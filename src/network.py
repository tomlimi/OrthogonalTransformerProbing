"""Classes for training and running inference on probes."""
import os

import tensorflow as tf
from abc import abstractmethod
from functools import partial

from tqdm import tqdm
import numpy as np

import constants
from data_support.tfrecord_wrapper import TFRecordReader

class Probe():

    ONPLATEU_DECAY = 0.1
    ES_PATIENCE = 4
    ES_DELTA = 1e-4

    def __init__(self, args):
        self.probe_rank = args.probe_rank
        self.model_dim = constants.MODEL_DIMS[args.bert_path]
        self.languages = args.languages
        self.tasks = args.tasks

        # self.bert_model = TFBertModel.from_pretrained(args.bert_path,
        #                                               output_hidden_states=True)

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

        self.optimal_loss = np.inf

        self._writer = tf.summary.create_file_writer(args.out_dir, flush_millis=10 * 1000, name='summary_writer')

    @abstractmethod
    def _forward(self, *args, **kwargs):
        pass

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

        # SRIP
        #
        # S = tf.transpose(w, perm=[1, 0]) @ w - tf.eye(w_cols)
        #
        # v = tf.random.uniform([w_cols, 1])
        # # small noise is added to solve the problem with orthogonal matrix at the begining
        # noise = tf.random.normal([w_cols, 1], 0.0, 1e-4)
        # u = S @ v + noise
        # v2 = S @ (u / tf.norm(u)) + noise
        # return tf.norm(v2)

    @abstractmethod
    def _loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def train_factory(self, *args, **kwargs):
        pass

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
    def data_pipeline(tf_data, languages, tasks, args, shuffle=True):

        datasets_to_interleve = []
        for lang in languages:
            for task in tasks:

                data = tf_data[lang][task]
                data = data.map(partial(Probe.decode, task=task, layer_idx=args.layer_index),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()

                if shuffle:
                    if 'der' in task:
                        data = data.repeat(8)
                    data = data.shuffle(constants.SHUFFLE_SIZE, args.seed)
                data = data.batch(args.batch_size)
                data = data.map(lambda *x: (lang, task, x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                data = data.prefetch(tf.data.experimental.AUTOTUNE)
                datasets_to_interleve.append(data)

        return tf.data.experimental.sample_from_datasets(datasets_to_interleve)

    def train(self, tf_reader, args):
        curr_patience = 0

        train = self.data_pipeline(tf_reader.train, self.languages, self.tasks, args)
        dev = {lang: {task: self.data_pipeline(tf_reader.dev, [lang], [task], args, shuffle=False)
                      for task in self.tasks}
               for lang in self.languages}
        
        for epoch_idx in range(args.epochs):
            
            progressbar = tqdm(enumerate(train))

            for batch_idx, (lang, task, batch) in progressbar:
                lang = lang.numpy().decode()
                task = task.numpy().decode()

                _, batch_target, batch_mask, batch_num_tokens, batch_embeddings = batch

                batch_loss = self._train_fns[lang][task](batch_target, batch_mask, batch_num_tokens, batch_embeddings)
                progressbar.set_description(f"Training, batch loss: {batch_loss:.4f}")

            eval_loss = self.evaluate(dev, 'validation', args)
            if eval_loss < self.optimal_loss - self.ES_DELTA:
                self.optimal_loss = eval_loss
                self.checkpoint_manager.save()
                curr_patience = 0
            else:
                curr_patience += 1

            if curr_patience > 0:
                self._lr *= self.ONPLATEU_DECAY
                self._optimizer.learning_rate.assign(self._lr)
                # zeroing optimizer weights in`no ml` setting to reproduce Hewitt's results
                if not self.ml_probe:
                    reset_variables = [np.zeros_like(var.numpy()) for var in self._optimizer.variables()]
                    self._optimizer.set_weights(reset_variables)
            if curr_patience > self.ES_PATIENCE:
                self.load(args)
                break
            with self._writer.as_default():
                tf.summary.scalar("train/learning_rate", self._optimizer.learning_rate)

    def evaluate(self, data, data_name, args):
        all_losses = np.zeros((len(self.languages), len(self.tasks)))
        for lang_idx, language in enumerate(self.languages):
            for task_idx, task in enumerate(self.tasks):

                progressbar = tqdm(enumerate(data[language][task]))
                for batch_idx, (lang2, task2, batch) in progressbar:
                    assert language == lang2.numpy().decode()
                    assert task == task2.numpy().decode()

                    _, batch_target, batch_mask, batch_num_tokens, batch_embeddings = batch
                    batch_loss = self.evaluate_on_batch(batch_target, batch_mask, batch_num_tokens, batch_embeddings,
                                                        language, task)
                    progressbar.set_description(f"Evaluating on {language} {task} loss: {batch_loss:.4f}")
                    all_losses[lang_idx][task_idx] += batch_loss

                all_losses[lang_idx][task_idx] = all_losses[lang_idx][task_idx] / (batch_idx + 1.)
                with self._writer.as_default():
                    tf.summary.scalar("{}/loss_{}_{}".format(data_name, language, task), all_losses[lang_idx][task_idx])

                print(f'{data_name} loss on {language} {task} : {all_losses[lang_idx][task_idx]:.4f}')
            
        with self._writer.as_default():
            tf.summary.scalar("{}/loss".format(data_name), all_losses.mean())
        return all_losses.mean()
        
    def load(self, args):
        self.checkpoint_manager.restore_or_initialize()


class DistanceProbe(Probe):
    """ Computes squared L2 distance after projection by a matrix.

    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """

    def __init__(self, args):
        print('Constructing DistanceProbe')
        super().__init__(args)
        
        if self._orthogonal_reg:
            # if orthogonalization is used multilingual probe is only diagonal scaling
            self.DistanceProbe = {task: tf.Variable(tf.random_uniform_initializer(minval=-0.5, maxval=0.5, seed=args.seed)
                                                    ((1, self.probe_rank,)),
                                                    trainable=True, name=f'{task}_probe', dtype=tf.float32)
                                  for task in self.tasks}
        else:
            self.DistanceProbe = {task: tf.Variable(tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=args.seed)
                                                    ((self.probe_rank, self.probe_rank)),
                                                    trainable=True, name=f'{task}_probe', dtype=tf.float32)
                                  for task in self.tasks}
        self._train_fns = {lang: {task: self.train_factory(lang, task)
                                  for task in self.tasks}
                           for lang in self.languages}
        
        #Checkpoint managment:
        self.ckpt = tf.train.Checkpoint(optimizer=self._optimizer, **self.DistanceProbe, **self.LanguageMaps)
        self.checkpoint_manager = tf.train.CheckpointManager(self.ckpt, os.path.join(args.out_dir, 'params'), max_to_keep=1)

    @tf.function
    def _forward(self, embeddings, max_token_len, language, task):
        """ Computes all n^2 pairs of distances after projection
        for each sentence in a batch.

        Note that due to padding, some distances will be non-zero for pads.
        Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j
        """

        embeddings = embeddings[:,:max_token_len,:]
        if self.ml_probe:
            embeddings = embeddings @ self.LanguageMaps[language]
        if self._orthogonal_reg:
            embeddings = embeddings * self.DistanceProbe[task]
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
                if self._orthogonal_reg and self.ml_probe:
                    ortho_penalty = self.ortho_reguralization(self.LanguageMaps[language])
                    loss += self._orthogonal_reg * ortho_penalty
                if self._l2_reg:
                    loss += self._l2_reg * tf.norm(self.LanguageMaps[language])
                    loss += self._l2_reg * tf.norm(self.DistanceProbe[task])

            if self.ml_probe:
                variables = [self.DistanceProbe[task], self.LanguageMaps[language]]
            else:
                variables = [self.DistanceProbe[task]]
            gradients = tape.gradient(loss, variables)
            gradient_norms = [tf.norm(grad) for grad in gradients]
            if self._clip_norm:
                gradients = [tf.clip_by_norm(grad, self._clip_norm) for grad in gradients]
            self._optimizer.apply_gradients(zip(gradients, variables))
            tf.summary.experimental.set_step(self._optimizer.iterations)

            with self._writer.as_default(), tf.summary.record_if(self._optimizer.iterations // len(self.languages) % 10 == 0):
                tf.summary.scalar("train/batch_loss_{}".format(language), loss)
                tf.summary.scalar("train/probe_gradient_norm", gradient_norms[0])
                if self._orthogonal_reg:
                    tf.summary.scalar("train/{}_nonorthogonality_penalty".format(language), ortho_penalty)
                if self.ml_probe:
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
    def predict_on_batch(self, token_len, embeddings, language, task):
        max_token_len = tf.reduce_max(token_len)
        predicted_distances = self._forward(embeddings, max_token_len, language, task)
        return predicted_distances


class DepthProbe(Probe):
    """ Computes squared L2 norm of words after projection by a matrix."""

    def __init__(self, args):
        print('Constructing DepthProbe')
        super().__init__(args)
        if self._orthogonal_reg:
            # if orthogonalization is used multilingual probe is only diagonal scaling
            self.DepthProbe = {task: tf.Variable(tf.random_uniform_initializer(minval=-0.5, maxval=0.5, seed=args.seed)
                                                    ((1, self.probe_rank,)),
                                                    trainable=True, name=f'{task}_probe', dtype=tf.float32)
                                  for task in self.tasks}
        else:
            self.DepthProbe = {task: tf.Variable(tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=args.seed)
                                                    ((self.probe_rank, self.probe_rank)),
                                                    trainable=True, name=f'{task}_probe', dtype=tf.float32)
                                  for task in self.tasks}
            
        self._train_fns = {lang: {task: self.train_factory(lang, task)
                                  for task in self.tasks}
                           for lang in self.languages}

        # Checkpoint managment:
        self.ckpt = tf.train.Checkpoint(optimizer=self._optimizer, **self.DepthProbe, **self.LanguageMaps)
        self.checkpoint_manager = tf.train.CheckpointManager(self.ckpt, os.path.join(args.out_dir, 'params'), max_to_keep=1)

    @tf.function
    def _forward(self, embeddings, max_token_len, language, task):
        """ Computes all n depths after projection for each sentence in a batch.
        Computes (Bh_i)^T(Bh_i) for all i
        """
        embeddings = embeddings[:, :max_token_len, :]
        if self.ml_probe:
            embeddings = embeddings @ self.LanguageMaps[language]
        if self._orthogonal_reg:
            embeddings = embeddings * self.DepthProbe[task]
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
                if self._orthogonal_reg and self.ml_probe:
                    ortho_penalty = self.ortho_reguralization(self.LanguageMaps[language])
                    loss += self._orthogonal_reg * ortho_penalty
                if self._l2_reg:
                    loss += self._l2_reg * tf.norm(self.LanguageMaps[language])
                    loss += self._l2_reg * tf.norm(self.DepthProbe[task])

            if self.ml_probe:
                variables = [self.DepthProbe[task], self.LanguageMaps[language]]
            else:
                variables = [self.DepthProbe[task]]
            gradients = tape.gradient(loss, variables)
            gradient_norms = [tf.norm(grad) for grad in gradients]
            if self._clip_norm:
                gradients = [tf.clip_by_norm(grad, self._clip_norm) for grad in gradients]
            self._optimizer.apply_gradients(zip(gradients, variables))
            tf.summary.experimental.set_step(self._optimizer.iterations)
            
            with self._writer.as_default(), tf.summary.record_if(self._optimizer.iterations // len(self.languages) % 10 == 0):
                tf.summary.scalar("train/batch_loss_{}".format(language), loss)
                tf.summary.scalar("train/probe_gradient_norm", gradient_norms[0])
                if self._orthogonal_reg:
                    tf.summary.scalar("train/{}_nonorthogonality_penalty".format(language), ortho_penalty)
                if self.ml_probe:
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
    def predict_on_batch(self, token_len, embeddings, language, task):
        max_token_len = tf.reduce_max(token_len)
        predicted_depths = self._forward(embeddings, max_token_len, language, task)
        return predicted_depths
