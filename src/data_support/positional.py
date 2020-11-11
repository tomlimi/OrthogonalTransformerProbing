import numpy as np
import tensorflow as tf

import constants
from data_support.conll_wrapper import ConllWrapper


class PositionalDistance(ConllWrapper):

	def __init__(self, conll_file, bert_tokenizer, lang='en'):
		super().__init__(conll_file, bert_tokenizer)

	def target_and_mask(self):
		"""Computes the distances between all pairs of words; returns them as a tensor.

		Returns:
		  target: A tensor of shape (num_examples, sentence_length, sentence_length) of positional distances.
		  mask: A tensor of shape (number of examples, sentence_length, sentence_length) specifying which elements of
		  the target should be used during training.
		"""
		seq_mask = tf.cast(tf.sequence_mask([len(sent_tokens) for sent_tokens in self.tokens], constants.MAX_TOKENS),
		                   tf.float32)
		seq_mask = tf.expand_dims(seq_mask, 1)
		seq_mask = seq_mask * tf.transpose(seq_mask, perm=[0, 2, 1])

		for sent_tokens, sentence_mask in zip(self.tokens, tf.unstack(seq_mask)):
			sentence_length = min(len(sent_tokens),
			                      constants.MAX_TOKENS)  # All observation fields must be of same length
			sentence_distances = np.zeros((constants.MAX_TOKENS, constants.MAX_TOKENS), dtype=np.float32)
			for i in range(sentence_length):
				for j in range(i+1, sentence_length):
					i_j_distance = j - i
					sentence_distances[i, j] = i_j_distance
					sentence_distances[j, i] = i_j_distance

			sentence_mask = tf.linalg.set_diag(sentence_mask, tf.repeat(0., constants.MAX_TOKENS))

			yield tf.constant(sentence_distances, dtype=tf.float32), sentence_mask


class PositionalDepth(ConllWrapper):

	def __init__(self, conll_file, bert_tokenizer, lang='en'):
		super().__init__(conll_file, bert_tokenizer)

	def target_and_mask(self):
		"""Computes the depth of each word; returns them as a tensor.

		Returns:
		  target: A tensor of shape (num_examples, sentence_length,) of depths in the parse tree.
		  mask: A tensor of shape (number of examples, sentence_length) specifying which elements of the target
		  should be used during training.
		"""
		seq_mask = tf.cast(tf.sequence_mask([len(sent_tokens) for sent_tokens in self.tokens], constants.MAX_TOKENS),
		                   tf.float32)

		for sent_tokens, sentence_mask in zip(self.tokens, tf.unstack(seq_mask)):
			sentence_length = min(len(sent_tokens),
			                      constants.MAX_TOKENS)  # All observation fields must be of same length
			sentence_depths = np.zeros(constants.MAX_TOKENS, dtype=np.float32)
			for i in range(sentence_length):
				sentence_depths[i] = float(i)

			yield tf.constant(sentence_depths, dtype=tf.float32), sentence_mask
