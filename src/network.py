"""Classes for training and running inference on probes."""
import os

import tensorflow as tf
from abc import abstractmethod
from functools import partial

from tqdm import tqdm
import numpy as np

import constants
from data_support.tfrecord_wrapper import TFRecordReader

# this should be read from data, only temporary solution
# TODO: get this info from tfrecord_wrapper
DER_SIZE = 1621
DER_TRAIN_SIZE = int(0.8 * DER_SIZE)
DER_DEV_SIZE = int(0.1 * DER_SIZE)


class Network():

    ONPLATEU_DECAY = 0.1
    ES_PATIENCE = 4
    ES_DELTA = 1e-4

    class Probe():
        def __init__(self, args):
            self.probe_rank = args.probe_rank
            self.model_dim = constants.MODEL_DIMS[args.bert_path]
            self.languages = args.languages

            self.ml_probe = args.ml_probe

            self.LanguageMaps = {lang: tf.Variable(tf.initializers.Identity(gain=1.0)((self.model_dim, self.probe_rank)),
                                               trainable=self.ml_probe, name='{}_map'.format(lang))
                             for lang in self.languages}

            self._layer_idx = args.layer_index
            self._lr = args.learning_rate
            self._clip_norm = args.clip_norm
            self._orthogonal_reg = args.ortho
            self._l2_reg = args.l2
            self._optimizer = tf.optimizers.Adam(lr=self._lr)

            self._writer = tf.summary.create_file_writer(args.out_dir, flush_millis=10 * 1000, name='summary_writer')


        @staticmethod
        @tf.function
        def ortho_reguralization(w):
            """DSO implementation according to:
            https://papers.nips.cc/paper/7680-can-we-gain-more-from-orthogonality-regularizations-in-training-deep-networks.pdf"""
            #
            w_cols = w.shape[0]
            w_rows = w.shape[1]
            reg = tf.norm(tf.transpose(w) @ w - tf.eye(w_cols)) + tf.norm(w @ tf.transpose(w) - tf.eye(w_rows))
            # to avoid NaN in gradient update
            if reg == 0:
                reg = 1e-6
            return reg

        def decrease_lr(self, decay_factor):
            self._lr *= decay_factor
            self._optimizer.learning_rate.assign(self._lr)
            # zeroing optimizer weights in`no ml` setting to reproduce Hewitt's results
            if not self.ml_probe:
                reset_variables = [np.zeros_like(var.numpy()) for var in self._optimizer.variables()]
                self._optimizer.set_weights(reset_variables)

    class DistanceProbe():
        """ Computes squared L2 distance after projection by a matrix.

        For a batch of sentences, computes all n^2 pairs of distances
        for each sentence in the batch.
        """

        def __init__(self, args, probe):
            print('Constructing DistanceProbe')
            self.probe = probe
            self.languages = args.languages
            self.tasks = [task for task in args.tasks if "distance" in task]

            if self.probe._orthogonal_reg:
                # if orthogonalization is used multilingual probe is only diagonal scaling
                self.DistanceProbe = {task: tf.Variable(tf.random_uniform_initializer(minval=-0.5, maxval=0.5, seed=args.seed)
                                                        ((1, self.probe.probe_rank,)),
                                                        trainable=True, name=f'{task}_probe', dtype=tf.float32)
                                      for task in self.tasks}
            else:
                self.DistanceProbe = {task: tf.Variable(tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=args.seed)
                                                        ((self.probe.probe_rank, self.probe.probe_rank)),
                                                        trainable=True, name=f'{task}_probe', dtype=tf.float32)
                                      for task in self.tasks}
            self._train_fns = {lang: {task: self.train_factory(lang, task)
                                      for task in self.tasks}
                               for lang in self.languages}

        @tf.function
        def _forward(self, embeddings, max_token_len, language, task, embeddings_gate=None):
            """ Computes all n^2 pairs of distances after projection
            for each sentence in a batch.

            Note that due to padding, some distances will be non-zero for pads.
            Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j
            """

            embeddings = embeddings[:,:max_token_len,:]
            if self.probe.ml_probe:
                embeddings = embeddings @ self.probe.LanguageMaps[language]
            if self.probe._orthogonal_reg:
                embeddings = embeddings * self.DistanceProbe[task]
                if embeddings_gate is not None:
                    embeddings = embeddings * embeddings_gate
            else:
                embeddings = embeddings @ self.DistanceProbe[task]


            embeddings = tf.expand_dims(embeddings, 1)  # shape [batch, 1, seq_len, emb_dim]
            transposed_embeddings = tf.transpose(embeddings, perm=(0, 2, 1, 3))  # shape [batch, seq_len, 1, emb_dim]
            diffs = embeddings - transposed_embeddings  # shape [batch, seq_len, seq_len, emb_dim]
            squared_diffs = tf.reduce_sum(tf.math.square(diffs), axis=-1) # shape [batch, seq_len, seq_len]
            return squared_diffs

        @tf.function
        def _loss(self, predicted_distances, gold_distances, mask, token_lens):
            sentence_loss = tf.reduce_sum(tf.abs(predicted_distances - gold_distances) * mask, axis=[1,2]) / \
                            tf.clip_by_value(tf.reduce_sum(mask, axis=[1,2]), 1., constants.MAX_TOKENS ** 2.)
            return tf.reduce_sum(sentence_loss)

        def train_factory(self, language, task):
            # separate train function is needed to avoid variable creation on non-first call
            # see: https://github.com/tensorflow/tensorflow/issues/27120
            @tf.function(experimental_relax_shapes=True)
            def train_on_batch(target, mask, token_len, embeddings):

                with tf.GradientTape() as tape:
                    max_token_len = tf.reduce_max(token_len)
                    target = target[:,:max_token_len,:max_token_len]
                    mask = mask[:,:max_token_len,:max_token_len]
                    predicted_distances = self._forward(embeddings, max_token_len, language, task)
                    loss = self._loss(predicted_distances, target, mask, token_len)
                    if self.probe._orthogonal_reg and self.probe.ml_probe:
                        ortho_penalty = self.probe.ortho_reguralization(self.probe.LanguageMaps[language])
                        loss += self.probe._orthogonal_reg * ortho_penalty
                    if self.probe._l2_reg:
                        loss += self.probe._l2_reg * tf.norm(self.probe.LanguageMaps[language])
                        loss += self.probe._l2_reg * tf.norm(self.DistanceProbe[task])

                if self.probe.ml_probe:
                    variables = [self.DistanceProbe[task], self.probe.LanguageMaps[language]]
                else:
                    variables = [self.DistanceProbe[task]]
                gradients = tape.gradient(loss, variables)
                gradient_norms = [tf.norm(grad) for grad in gradients]
                if self.probe._clip_norm:
                    gradients = [tf.clip_by_norm(grad, self.probe._clip_norm) for grad in gradients]
                self.probe._optimizer.apply_gradients(zip(gradients, variables))
                tf.summary.experimental.set_step(self.probe._optimizer.iterations)

                with self.probe._writer.as_default(), tf.summary.record_if(self.probe._optimizer.iterations % 10 == 0):
                    tf.summary.scalar("train/batch_loss_{}".format(language), loss)
                    tf.summary.scalar("train/probe_gradient_norm", gradient_norms[0])
                    if self.probe._orthogonal_reg:
                        tf.summary.scalar("train/{}_nonorthogonality_penalty".format(language), ortho_penalty)
                    if self.probe.ml_probe:
                        tf.summary.scalar("train/{}_map_gradient_norm".format(language), gradient_norms[1])

                return loss
            return train_on_batch

        @tf.function(experimental_relax_shapes=True)
        def evaluate_on_batch(self, target, mask, token_len, embeddings, language, task):
            max_token_len = tf.reduce_max(token_len)
            target = target[:, :max_token_len, :max_token_len]
            mask = mask[:, :max_token_len, :max_token_len]
            predicted_distances = self._forward(embeddings, max_token_len, language, task)
            loss = self._loss(predicted_distances, target, mask, token_len)
            return loss

        @tf.function(experimental_relax_shapes=True)
        def predict_on_batch(self, token_len, embeddings, language, task, embeddings_gate=None):
            max_token_len = tf.reduce_max(token_len)
            predicted_distances = self._forward(embeddings, max_token_len, language, task, embeddings_gate)
            return predicted_distances


    class DepthProbe():
        """ Computes squared L2 norm of words after projection by a matrix."""

        def __init__(self, args, probe):
            print('Constructing DepthProbe')
            self.probe = probe
            self.languages = args.languages
            self.tasks = [task for task in args.tasks if "depth" in task]

            if self.probe._orthogonal_reg:
                # if orthogonalization is used multilingual probe is only diagonal scaling
                self.DepthProbe = {task: tf.Variable(tf.random_uniform_initializer(minval=-0.5, maxval=0.5, seed=args.seed)
                                                        ((1, self.probe.probe_rank,)),
                                                        trainable=True, name=f'{task}_probe', dtype=tf.float32)
                                      for task in self.tasks}
            else:
                self.DepthProbe = {task: tf.Variable(tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=args.seed)
                                                        ((self.probe.probe_rank, self.probe.probe_rank)),
                                                        trainable=True, name=f'{task}_probe', dtype=tf.float32)
                                      for task in self.tasks}

            self._train_fns = {lang: {task: self.train_factory(lang, task)
                                      for task in self.tasks}
                               for lang in self.languages}

        @tf.function
        def _forward(self, embeddings, max_token_len, language, task, embeddings_gate=None):
            """ Computes all n depths after projection for each sentence in a batch.
            Computes (Bh_i)^T(Bh_i) for all i
            """
            embeddings = embeddings[:, :max_token_len, :]
            if self.probe.ml_probe:
                embeddings = embeddings @ self.probe.LanguageMaps[language]
            if self.probe._orthogonal_reg:
                embeddings = embeddings * self.DepthProbe[task]
                if embeddings_gate is not None:
                    embeddings = embeddings * embeddings_gate
            else:
                embeddings = embeddings @ self.DepthProbe[task]

            squared_norms = tf.norm(embeddings, ord='euclidean', axis=2) ** 2
            return squared_norms

        @tf.function
        def _loss(self, predicted_depths, gold_depths, mask, token_lens):
            sentence_loss = tf.reduce_sum(tf.abs(predicted_depths - gold_depths) * mask, axis=1) / \
                            tf.clip_by_value(tf.reduce_sum(mask, axis=1), 1., constants.MAX_TOKENS)
            return tf.reduce_sum(sentence_loss)

        def train_factory(self, language, task):
            @tf.function(experimental_relax_shapes=True)
            def train_on_batch(target, mask, token_len, embeddings):

                with tf.GradientTape() as tape:
                    max_token_len = tf.reduce_max(token_len)
                    target = target[:,:max_token_len]
                    mask = mask[:,:max_token_len]
                    predicted_depths = self._forward(embeddings, max_token_len, language, task)
                    loss = self._loss(predicted_depths, target, mask, token_len)
                    if self.probe._orthogonal_reg and self.probe.ml_probe:
                        ortho_penalty = self.probe.ortho_reguralization(self.probe.LanguageMaps[language])
                        loss += self.probe._orthogonal_reg * ortho_penalty
                    if self.probe._l2_reg:
                        loss += self.probe._l2_reg * tf.norm(self.probe.LanguageMaps[language])
                        loss += self.probe._l2_reg * tf.norm(self.DepthProbe[task])

                if self.probe.ml_probe:
                    variables = [self.DepthProbe[task], self.probe.LanguageMaps[language]]
                else:
                    variables = [self.DepthProbe[task]]
                gradients = tape.gradient(loss, variables)
                gradient_norms = [tf.norm(grad) for grad in gradients]
                if self.probe._clip_norm:
                    gradients = [tf.clip_by_norm(grad, self.probe._clip_norm) for grad in gradients]
                self.probe._optimizer.apply_gradients(zip(gradients, variables))
                tf.summary.experimental.set_step(self.probe._optimizer.iterations)

                with self.probe._writer.as_default(), tf.summary.record_if(self.probe._optimizer.iterations % 10 == 0):
                    tf.summary.scalar("train/batch_loss_{}".format(language), loss)
                    tf.summary.scalar("train/probe_gradient_norm", gradient_norms[0])
                    if self.probe._orthogonal_reg:
                        tf.summary.scalar("train/{}_nonorthogonality_penalty".format(language), ortho_penalty)
                    if self.probe.ml_probe:
                        tf.summary.scalar("train/{}_map_gradient_norm".format(language), gradient_norms[1])

                return loss
            return train_on_batch

        @tf.function(experimental_relax_shapes=True)
        def evaluate_on_batch(self, target, mask, token_len, embeddings, language, task):
            max_token_len = tf.reduce_max(token_len)
            target = target[:, :max_token_len]
            mask = mask[:, :max_token_len]
            predicted_depths = self._forward(embeddings, max_token_len, language, task)
            loss = self._loss(predicted_depths, target, mask, token_len)
            return loss

        @tf.function(experimental_relax_shapes=True)
        def predict_on_batch(self, token_len, embeddings, language, task, embeddings_gate=None):
            max_token_len = tf.reduce_max(token_len)
            predicted_depths = self._forward(embeddings, max_token_len, language, task, embeddings_gate)
            return predicted_depths

    def __init__(self, args):

        self.languages = args.languages
        self.tasks = args.tasks

        self.probe = self.Probe(args)
        self.depth_probe = self.DepthProbe(args, self.probe)
        self.distance_probe = self.DistanceProbe(args, self.probe)

        self.optimal_loss = np.inf

        # Checkpoint managment:
        self.ckpt = tf.train.Checkpoint(optimizer=self.probe._optimizer, **self.probe.LanguageMaps,
                                        **self.distance_probe.DistanceProbe,
                                        **self.depth_probe.DepthProbe)
        self.checkpoint_manager = tf.train.CheckpointManager(self.ckpt, os.path.join(args.out_dir, 'params'),
                                                             max_to_keep=1)

    @staticmethod
    def decode(serialized_example, task, layer_idx):

        x = tf.io.parse_example(
            serialized_example,
            features={
                'num_tokens': tf.io.FixedLenFeature([], tf.int64),
                'index': tf.io.FixedLenFeature([], tf.int64),
                f"layer_{layer_idx}": tf.io.FixedLenFeature([], tf.string),
                f'target_{task}': tf.io.FixedLenFeature([], tf.string),
                f'mask_{task}': tf.io.FixedLenFeature([], tf.string)
            })

        index = tf.cast(x["index"], dtype=tf.int64)
        target = tf.io.parse_tensor(x[f"target_{task}"], out_type=tf.float32)
        mask = tf.io.parse_tensor(x[f"mask_{task}"], out_type=tf.float32)
        num_tokens = tf.cast(x["num_tokens"], dtype=tf.int64)
        embeddings = tf.io.parse_tensor(x[f"layer_{layer_idx}"], out_type=tf.float32)

        return index, target, mask, num_tokens, embeddings

    @staticmethod
    def data_pipeline(tf_data, languages, tasks, args, mode='train'):

        datasets_to_interleve = []
        for lang in languages:
            if lang not in tf_data:
                raise ValueError(f"Language: {lang} not found in the data set")
            for task in tasks:
                if task not in tf_data[lang]:
                    raise ValueError(f"Task: {task} not found in the data set")

                data = tf_data[lang][task]

                data = data.map(partial(Network.decode, task=task, layer_idx=args.layer_index),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()

                if mode == 'train':
                    data = data.shuffle(constants.SHUFFLE_SIZE, args.seed)
                data = data.batch(args.batch_size)
                data = data.map(lambda *x: (lang, task, x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                data = data.prefetch(tf.data.experimental.AUTOTUNE)
                datasets_to_interleve.append(data)

        return tf.data.experimental.sample_from_datasets(datasets_to_interleve)

    def train(self, tf_reader, args):
        curr_patience = 0

        train = self.data_pipeline(tf_reader.train, self.languages, self.tasks, args, mode='train')
        dev = {lang: {task: self.data_pipeline(tf_reader.dev, [lang], [task], args, mode='dev')
                      for task in self.tasks}
               for lang in self.languages}

        for epoch_idx in range(args.epochs):

            progressbar = tqdm(enumerate(train))

            for batch_idx, (lang, task, batch) in progressbar:
                lang = lang.numpy().decode()
                task = task.numpy().decode()

                _, batch_target, batch_mask, batch_num_tokens, batch_embeddings = batch

                if 'depth' in task:
                    batch_loss = self.depth_probe._train_fns[lang][task](
                        batch_target, batch_mask, batch_num_tokens, batch_embeddings)
                elif 'distance' in task:
                    batch_loss = self.distance_probe._train_fns[lang][task](
                        batch_target, batch_mask, batch_num_tokens, batch_embeddings)

                progressbar.set_description(f"Training, batch loss: {batch_loss:.4f}")

            eval_loss = self.evaluate(dev, 'validation', args)
            if eval_loss < self.optimal_loss - self.ES_DELTA:
                self.optimal_loss = eval_loss
                self.checkpoint_manager.save()
                curr_patience = 0
            else:
                curr_patience += 1

            if curr_patience > 0:
                self.probe.decrease_lr(self.ONPLATEU_DECAY)
            if curr_patience > self.ES_PATIENCE:
                self.load(args)
                break
            with self.probe._writer.as_default():
                tf.summary.scalar("train/learning_rate", self.probe._optimizer.learning_rate)

    def evaluate(self, data, data_name, args):
        all_losses = np.zeros((len(self.languages), len(self.tasks)))
        for lang_idx, language in enumerate(self.languages):
            for task_idx, task in enumerate(self.tasks):

                progressbar = tqdm(enumerate(data[language][task]))
                for batch_idx, (_, _, batch) in progressbar:
                    _, batch_target, batch_mask, batch_num_tokens, batch_embeddings = batch
                    if "distance" in task:
                        batch_loss = self.distance_probe.evaluate_on_batch(batch_target, batch_mask, batch_num_tokens,
                                                                           batch_embeddings, language, task)
                    elif "depth" in task:
                        batch_loss = self.depth_probe.evaluate_on_batch(batch_target, batch_mask, batch_num_tokens,
                                                                           batch_embeddings, language, task)
                    progressbar.set_description(f"Evaluating on {language} {task} loss: {batch_loss:.4f}")
                    all_losses[lang_idx][task_idx] += batch_loss

                all_losses[lang_idx][task_idx] = all_losses[lang_idx][task_idx] / (batch_idx + 1.)
                with self.probe._writer.as_default():
                    tf.summary.scalar("{}/loss_{}_{}".format(data_name, language, task), all_losses[lang_idx][task_idx])

                print(f'{data_name} loss on {language} {task} : {all_losses[lang_idx][task_idx]:.4f}')

        with self.probe._writer.as_default():
            tf.summary.scalar("{}/loss".format(data_name), all_losses.mean())
        return all_losses.mean()

    def load(self, args):
        self.checkpoint_manager.restore_or_initialize()
