"""Classes for training and running inference on probes."""
import os
import sys

import tensorflow as tf
import bert
import tensorflow_hub as hub
from abc import abstractmethod

from tqdm import tqdm
import numpy as np

#from bert_wrapper import BertWrapper()

class BertModel():
    
    def __init__(self, modelBertDir, language, size="base", casing="uncased", layer_idx=-1):
        self.max_seq_length = 128
        bertDir = os.path.join(modelBertDir, "{}-{}-{}".format(language, size, casing))
        if not os.path.exists(bertDir):
            raise ValueError(
                "The requested Bert model combination {}-{}-{} does not exist".format(language, size, casing))
    
        bert_params = bert.params_from_pretrained_ckpt(bertDir)
    
        self.bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert", out_layer_ndxs=[layer_idx])

        self.bert_layer.apply_adapter_freeze()

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.max_seq_length,), dtype='int32', name='input_ids'),
            self.bert_layer])

        self.model.build(input_shape=(None, self.max_seq_length))
        
        checkpointName = os.path.join(bertDir, "bert_model.ckpt")
        #
        bert.load_stock_weights(self.bert_layer, checkpointName)
        
        vocab_file = os.path.join(bertDir, "vocab.txt")
        do_lower_case = (casing == "uncased")
        self.tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

    def __call__(self, wordpiece_ids):
        embeddings = self.model(wordpiece_ids)
        return embeddings


class Probe():

    ES_PATIENCE = 5
    ES_DELTA = 1e-4

    @abstractmethod
    def __init__(self, args):
        pass


class DistanceProbe(Probe):
    """ Computes squared L2 distance after projection by a matrix.

    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """

    def __init__(self, args):
        print('Constructing DistanceProbe')
        super(DistanceProbe, self).__init__()
        self.probe_rank = args.probe_ran
        self.model_dim = args.bert_dim
        self.languages = args.languages

        self.bert_model = BertModel(args.bert_dir, "multilingual", layer_idx=args.layer_index)
        self.Distance_Probe = tf.Variable(tf.initializers.GlorotUniform(seed=42)((self.probe_rank, self.model_dim )),
                                          trainable=True, name='distance_probe')

        self.Language_Maps = {lang: tf.Variable(tf.eye(self.model_dim), trainable=True, name='{}_map'.format(lang))
                              for lang in args.languages}

        self._optimizer = tf.optimizers.Adam()

    @tf.function
    def forward(self, batch):
        """ Computes all n^2 pairs of distances after projection
        for each sentence in a batch.

        Note that due to padding, some distances will be non-zero for pads.
        Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j
        """
        embeddings = self.bert_model(batch.wordpieces)
        print(embeddings.shape)
        # average wordpieces to obtain word representation
        # cut to max nummber of words in batch
        embeddings = tf.map_fn(tf.math.unsorted_segment_mean, [embeddings, batch.segments, batch.max_token_len])
        embeddings = self.Language_Maps[batch.language] @ embeddings
        embeddings = self.Distance_Probe @ embeddings
        #embeddings_shape = tf.shape(embeddings)
        embeddings = tf.expand_dims(embeddings, 1)  # shape [batch, 1, seq_len, emb_dim]
        transposed_embeddings = tf.transpose(embeddings, perm=(1, 2))  # shape [batch, seq_len, 1, emb_dim]
        diffs = embeddings - transposed_embeddings  # shape [batch, seq_len, seq_len, emb_dim]
        squared_diffs = tf.reduce_sum(tf.math.square(diffs), axis=-1) # shape [batch, seq_len, seq_len]
        return squared_diffs

    @tf.function
    def _loss(self, predicted_distances, gold_distances, token_lens):
        sentence_loss = tf.reduce_sum(tf.abs(predicted_distances - gold_distances), axis=[1,2]) / (token_lens ** 2)
        return tf.reduce_mean(sentence_loss)

    @tf.function
    def train_on_batch(self, batch):

        with tf.GradientTape() as tape:
            predicted_distances = self.forward(batch)
            loss = self._loss(predicted_distances, batch.target, batch.token_len)

        variables = [self.Distance_Probe, self.Language_Maps[batch.language]]
        gradients = tape.gradient(loss, variables)
        self._optimizer.apply_gradients(zip(gradients, variables))

        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default(), tf.summary.record_if(self._optimizer.iterations % 100 == 0):
            for metric_name, metric in self._metrics.items():
                tf.summary.scalar("train/" + metric_name, metric.result())

        return loss

    def train(self, dep_dataset, args):
        curr_patience = 0
        best_weights = None

        for epoch_idx in range(args.epochs):
            for train_batch in tqdm(dep_dataset.train.train_batches(args.batch_size)):
                batch_loss = self.train_batch(train_batch)

            #TODO: method to save variables/network
            # eval_loss = self.evaluate(dep_dataset.dev, 'validation', args)
            #
            # if eval_loss < self.lowest_loss - self.ES_DELTA:
            #     self.lowest_loss = eval_loss
            #     best_weights = self.model.get_weights()
            #     curr_patience = 0
            # else:
            #     curr_patience += 1
            #
            # if curr_patience > self.ES_PATIENCE:
            #     self.model.set_weights(best_weights)
            #     break

    def evaluate_batch(self, batch):
        predicted_distances = self.forward(batch)
        loss = self._loss(predicted_distances, batch.target, batch.token_len)
        return loss

    def evaluate(self, data, data_name, args):
        all_losses = np.zeros((len(self.languages)))
        for lang_idx, language in self.languages:
            progressbar = tqdm(data.evaluate_batches(args.batch_size))
            for batch_idx, batch in enumerate(progressbar):
                batch_loss = self.evaluate_batch(batch)
                progressbar.set_description(f"Evaluating on {language}! loss: {batch_loss}")

                all_losses[lang_idx] += batch_loss

            all_losses[lang_idx] = all_losses[lang_idx] / (batch_idx + 1)

        return all_losses.mean()


class OneWordPSDProbe(Probe):
    """ Computes squared L2 norm of words after projection by a matrix."""

    # def __init__(self, args):
    #     print('Constructing OneWordPSDProbe')
    #     super(OneWordPSDProbe, self).__init__()
    #     self.args = args
    #     self.probe_rank = args['probe']['maximum_rank']
    #     self.model_dim = args['model']['hidden_dim']
    #     self.proj = tf.nn.Parameter(data=torch.zeros(self.model_dim, self.probe_rank))
    #     tf.nn.init.uniform_(self.proj, -0.05, 0.05)
    #     self.to(args['device'])
    #
    # def forward(self, batch):
    #     """ Computes all n depths after projection
    #     for each sentence in a batch.
    #
    #     Computes (Bh_i)^T(Bh_i) for all i
    #
    #     Args:
    #       batch: a batch of word representations of the shape
    #         (batch_size, max_seq_len, representation_dim)
    #     Returns:
    #       A tensor of depths of shape (batch_size, max_seq_len)
    #     """
    #     transformed = torch.matmul(batch, self.proj)
    #     batchlen, seqlen, rank = transformed.size()
    #     norms = torch.bmm(transformed.view(batchlen * seqlen, 1, rank),
    #                       transformed.view(batchlen * seqlen, rank, 1))
    #     norms = norms.view(batchlen, seqlen)
    #     return norms




