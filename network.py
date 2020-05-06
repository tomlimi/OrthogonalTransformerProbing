"""Classes for training and running inference on probes."""
import os
import sys

import tensorflow as tf
import bert
import tensorflow_hub as hub

from tqdm import tqdm

#from bert_wrapper import BertWrapper()

class BertWrapper():
    
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


    def get_ids(self, tokens):
        """Token ids from Tokenizer vocab"""
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = token_ids + [0] * (self.max_seq_length - len(token_ids))
        return input_ids
    
    def __call__(self, sentences):
        input = tf.constant([self.get_ids((["[CLS]"] + self.tokenizer.tokenize(sentence) + ["[SEP]"])) for sentence in sentences])
        print(input)
        all_embs = self.model(input)
        return all_embs


class Probe():
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

        self.bert_wrapper = BertWrapper("multilingual",args.layer_index)
        self.Distance_Probe = tf.Variable(tf.initializers.GlorotUniform(seed=42)((self.probe_rank, self.model_dim )),
                                          trainable=True, name='distance_probe')

        self.Language_Maps = {lang: tf.Variable(tf.eye(self.model_dim), trainable=True, name='{}_map'.format(lang))
                              for lang in args.languages}

        self._optimizer = tf.optimizers.Adam()

    @tf.function
    def train_on_batch(self, batch, language):
        """ Computes all n^2 pairs of distances after projection
        for each sentence in a batch.

        Note that due to padding, some distances will be non-zero for pads.
        Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j

        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
        """

        with tf.GradientTape() as tape:
            predicted_distances = self.forward(batch["features"], language)
            loss = self._loss(predicted_distances, batch["distances"])

    @tf.function
    def forward(self, features, language):
        embeddings = self.bert_wrapper.model_fn(features, tf.estimator.ModeKeys.PREDICT)
        embeddings = self.Language_Maps[language] @ embeddings
        embeddings = self.Distance_Probe @ embeddings
        # batchlen, seqlen, rank \
        embeddings_shape = tf.shape(embeddings)
        embeddings = tf.expand_dims(embeddings,1)
        # embeddings = transformed.expand(-1, -1, seqlen, -1)
        transposed_embeddings = tf.transpose(embeddings, perm=(1, 2))
        diffs = embeddings - transposed_embeddings
        squared_diffs = tf.reduce_sum(tf.math.square(diffs), axis=-1)
        return squared_diffs

    @tf.function
    def _loss(self, predicted_distances, ):

        pass
    
    # @tf.function
    # def forward(self, batch, language):
    #     """ Computes all n^2 pairs of distances after projection
    #     for each sentence in a batch.
    #
    #     Note that due to padding, some distances will be non-zero for pads.
    #     Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j
    #
    #     Args:
    #       batch: a batch of word representations of the shape
    #         (batch_size, max_seq_len, representation_dim)
    #     Returns:
    #       A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
    #     """
    #
    #
    #     transformed = self.Distance_Probe @ self.Language_Maps[language] @ batch
    #     batchlen, seqlen, rank = transformed.size()
    #     transformed = transformed.unsqueeze(2)
    #     transformed = transformed.expand(-1, -1, seqlen, -1)
    #     transposed = transformed.transpose(1, 2)
    #     diffs = transformed - transposed
    #     squared_diffs = diffs.pow(2)
    #     squared_distances = torch.sum(squared_diffs, -1)
    #     return squared_distances


class OneWordPSDProbe(Probe):
    """ Computes squared L2 norm of words after projection by a matrix."""

    def __init__(self, args):
        print('Constructing OneWordPSDProbe')
        super(OneWordPSDProbe, self).__init__()
        self.args = args
        self.probe_rank = args['probe']['maximum_rank']
        self.model_dim = args['model']['hidden_dim']
        self.proj = tf.nn.Parameter(data=torch.zeros(self.model_dim, self.probe_rank))
        tf.nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(args['device'])

    def forward(self, batch):
        """ Computes all n depths after projection
        for each sentence in a batch.

        Computes (Bh_i)^T(Bh_i) for all i

        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of depths of shape (batch_size, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = transformed.size()
        norms = torch.bmm(transformed.view(batchlen * seqlen, 1, rank),
                          transformed.view(batchlen * seqlen, rank, 1))
        norms = norms.view(batchlen, seqlen)
        return norms




