"""Classes for training and running inference on probes."""
import os

import tensorflow as tf
import bert
from abc import abstractmethod

from tqdm import tqdm
import numpy as np

import constants

class BertModel():

    def __init__(self, modelBertDir, layer_idx=-1):

        bert_params = bert.params_from_pretrained_ckpt(modelBertDir)
        self.bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert", out_layer_ndxs=[layer_idx])
        self.bert_layer.apply_adapter_freeze()

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(constants.MAX_WORDPIECES,), dtype='int32', name='input_ids'),
            self.bert_layer])

        self.model.build(input_shape=(None, constants.MAX_WORDPIECES))

        checkpointName = os.path.join(modelBertDir, "bert_model.ckpt")
        bert.load_stock_weights(self.bert_layer, checkpointName)

    def __call__(self, wordpiece_ids):
        embeddings = self.model(wordpiece_ids)
        return embeddings


class Probe():

    ONPLATEU_DECAY = 0.1
    ES_PATIENCE = 4
    ES_DELTA = 1e-4

    def __init__(self, args):
        self.probe_rank = args.probe_rank
        self.model_dim = args.bert_dim
        self.languages = args.train_languages

        self.bert_model = BertModel(args.bert_dir, layer_idx=args.layer_index)

        self.ml_probe = args.ml_probe

        self.LanguageMaps = {lang: tf.Variable(tf.initializers.Identity(gain=1.0)((self.model_dim, self.model_dim)),
                                               trainable=self.ml_probe, name='{}_map'.format(lang))
                             for lang in self.languages}

        self._lr = args.learning_rate
        self._optimizer = tf.optimizers.Adam(lr=self._lr)

        self.optimal_loss = np.inf

    @abstractmethod
    def _forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def _loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def train_factory(self, *args, **kwargs):
        pass

    def train(self, dep_dataset, args):
        curr_patience = 0
        for epoch_idx in range(args.epochs):
            progressbar = tqdm(enumerate(dep_dataset.train.train_batches(args.batch_size)))
            for batch_idx, batch in progressbar:
                batch_loss = self._train_fns[batch.language](batch.wordpieces, batch.segments, batch.token_len,
                                                             batch.max_token_len, batch.target)
                progressbar.set_description(f"Training, batch loss: {batch_loss:.4f}")

            eval_loss = self.evaluate(dep_dataset.dev, 'validation', args)
            if eval_loss < self.optimal_loss - self.ES_DELTA:
                self.optimal_loss = eval_loss
                self.checkpoint_manager.save()
                curr_patience = 0
            else:
                curr_patience += 1

            if curr_patience > 0:
                self._lr *= self.ONPLATEU_DECAY
                self._optimizer.learning_rate.assign(self._lr)
                reset_variables = [np.zeros_like(var.numpy()) for var in self._optimizer.variables()]
                self._optimizer.set_weights(reset_variables)
            if curr_patience > self.ES_PATIENCE:
                self.load(args)
                break
            print(self._optimizer.get_config())

    def evaluate(self, data, data_name, args):
        all_losses = np.zeros((len(self.languages)))
        for lang_idx, language in enumerate(self.languages):
            progressbar = tqdm(enumerate(data.evaluate_batches(language, args.batch_size)))
            for batch_idx, batch in progressbar:
                batch_loss = self.evaluate_on_batch(batch.wordpieces, batch.segments, batch.token_len,
                                                    batch.max_token_len, batch.language, batch.target)
                progressbar.set_description(f"Evaluating on {language}, loss: {batch_loss:.4f}")
                all_losses[lang_idx] += batch_loss

            all_losses[lang_idx] = all_losses[lang_idx] / (batch_idx + 1.)
            print(f'Evaluation loss for: {language} : {all_losses[lang_idx]:.4f}')

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

        self.DistanceProbe = tf.Variable(tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=args.seed)
                                         ((self.probe_rank, self.model_dim)),
                                         trainable=True, name='distance_probe', dtype=tf.float32)
        self._train_fns = {lang: self.train_factory(lang) for lang in self.languages}
        
        #Checkpoint managment:
        self.ckpt = tf.train.Checkpoint(optimizer=self._optimizer, distance_probe=self.DistanceProbe, **self.LanguageMaps)
        self.checkpoint_manager = tf.train.CheckpointManager(self.ckpt, os.path.join(args.out_dir, 'params'), max_to_keep=1)

    @tf.function
    def _forward(self, wordpieces, segments, max_token_len, language):
        """ Computes all n^2 pairs of distances after projection
        for each sentence in a batch.

        Note that due to padding, some distances will be non-zero for pads.
        Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j
        """
        embeddings = self.bert_model(wordpieces)
        # average wordpieces to obtain word representation
        # cut to max nummber of words in batch, note that batch.max_token_len is a tensor, bu all the values are the same
        embeddings = tf.map_fn(lambda x: tf.math.unsorted_segment_mean(x[0], x[1], x[2]),
                               (embeddings, segments, max_token_len), dtype=tf.float32)
        #embeddings = tf.reshape(embeddings, [embeddings.shape[0], max_token_len[0], embeddings.shape[2]])
        if self.ml_probe:
            embeddings = embeddings @ self.LanguageMaps[language]
        embeddings = embeddings @ self.DistanceProbe
        embeddings = tf.expand_dims(embeddings, 1)  # shape [batch, 1, seq_len, emb_dim]
        transposed_embeddings = tf.transpose(embeddings, perm=(0, 2, 1, 3))  # shape [batch, seq_len, 1, emb_dim]
        diffs = embeddings - transposed_embeddings  # shape [batch, seq_len, seq_len, emb_dim]
        squared_diffs = tf.reduce_sum(tf.math.square(diffs), axis=-1) # shape [batch, seq_len, seq_len]
        return squared_diffs

    @tf.function
    def _loss(self, predicted_distances, gold_distances, token_lens):
        sentence_loss = tf.reduce_sum(tf.abs(predicted_distances - gold_distances), axis=[1,2]) / (tf.cast(token_lens, dtype=tf.float32) ** 2)
        return tf.reduce_sum(sentence_loss)

    def train_factory(self,language):
        # separate train function is needed to avoid variable creation on non-first call
        # see: https://github.com/tensorflow/tensorflow/issues/27120
        @tf.function(experimental_relax_shapes=True)
        def train_on_batch(wordpieces, segments, token_len, max_token_len, target):

            with tf.GradientTape() as tape:
                predicted_distances = self._forward(wordpieces, segments, max_token_len, language)
                loss = self._loss(predicted_distances, target, token_len)

            if self.ml_probe:
                variables = [self.DistanceProbe, self.LanguageMaps[language]]
            else:
                variables = [self.DistanceProbe]
            gradients = tape.gradient(loss, variables)
            self._optimizer.apply_gradients(zip(gradients, variables))

            return loss
        return train_on_batch

    @tf.function(experimental_relax_shapes=True)
    def evaluate_on_batch(self, wordpieces, segments, token_len, max_token_len, language, target):
        predicted_distances = self._forward(wordpieces, segments, max_token_len, language)
        loss = self._loss(predicted_distances, target, token_len)
        return loss
    
    @tf.function(experimental_relax_shapes=True)
    def predict_on_batch(self, wordpieces, segments, token_len, max_token_len, language):
        predicted_distances = self._forward(wordpieces, segments, max_token_len, language)
        return predicted_distances


class DepthProbe(Probe):
    """ Computes squared L2 norm of words after projection by a matrix."""

    def __init__(self, args):
        print('Constructing DepthProbe')
        super().__init__(args)
        self.DepthProbe = tf.Variable(tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=args.seed)
                                      ((self.probe_rank, self.model_dim)),
                                      trainable=True, name='depth_probe', dtype=tf.float32)
        self._train_fns = {lang: self.train_factory(lang) for lang in self.languages}

        # Checkpoint managment:
        self.ckpt = tf.train.Checkpoint(optimizer=self._optimizer, depth_probe=self.DepthProbe, **self.LanguageMaps)
        self.checkpoint_manager = tf.train.CheckpointManager(self.ckpt, os.path.join(args.out_dir, 'params'), max_to_keep=1)

    @tf.function
    def _forward(self, wordpieces, segments, max_token_len, language):
        """ Computes all n depths after projection for each sentence in a batch.
        Computes (Bh_i)^T(Bh_i) for all i
        """
        embeddings = self.bert_model(wordpieces)
        embeddings = tf.map_fn(lambda x: tf.math.unsorted_segment_mean(x[0], x[1], x[2]),
                               (embeddings, segments, max_token_len), dtype=tf.float32)
        #embeddings = tf.reshape(embeddings, [embeddings.shape[0], max_token_len[0], embeddings.shape[2]])
        if self.ml_probe:
            embeddings = embeddings @ self.LanguageMaps[language]
        embeddings = embeddings @ self.DepthProbe

        squared_norms = tf.norm(embeddings, ord='euclidean', axis=2) ** 2
        return squared_norms

    @tf.function
    def _loss(self, predicted_depths, gold_depths, token_lens):
        sentence_loss = tf.reduce_sum(tf.abs(predicted_depths - gold_depths), axis=1) / (tf.cast(token_lens, dtype=tf.float32))
        return tf.reduce_sum(sentence_loss)

    def train_factory(self,language):
        @tf.function(experimental_relax_shapes=True)
        def train_on_batch(wordpieces, segments, token_len, max_token_len, target):

            with tf.GradientTape() as tape:
                predicted_depths = self._forward(wordpieces, segments, max_token_len, language)
                loss = self._loss(predicted_depths, target, token_len)

            if self.ml_probe:
                variables = [self.DepthProbe, self.LanguageMaps[language]]
            else:
                variables = [self.DepthProbe]
            gradients = tape.gradient(loss, variables)
            self._optimizer.apply_gradients(zip(gradients, variables))

            return loss
        return train_on_batch

    @tf.function(experimental_relax_shapes=True)
    def evaluate_on_batch(self, wordpieces, segments, token_len, max_token_len, language, target):
        predicted_depths = self._forward(wordpieces, segments, max_token_len, language)
        loss = self._loss(predicted_depths, target, token_len)
        return loss
    
    @tf.function(experimental_relax_shapes=True)
    def predict_on_batch(self, wordpieces, segments, token_len, max_token_len, language):
        predicted_depths = self._forward(wordpieces, segments, max_token_len, language)
        return predicted_depths
