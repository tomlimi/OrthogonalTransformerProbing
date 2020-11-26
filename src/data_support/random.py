import numpy as np
import tensorflow as tf

import constants
from data_support.dependency import DependencyDistance, DependencyDepth


class RandomDistance(DependencyDistance):

	max_wordpieces = constants.MAX_WORDPIECES

	def __init__(self, conll_file, bert_tokenizer, lang='en'):
		super().__init__(conll_file, bert_tokenizer)


	def target_and_mask(self):
		"""Computes the distances between all pairs of words in a random tree; returns them as a tensor.

		Returns:
		  target: A tensor of shape (num_examples, sentence_length, sentence_length) of distances
		  in the parse tree.
		  mask: A tensor of shape (number of examples, sentence_length, sentence_length) specifying which elements of
		  the target should be used during training.
		"""
		seq_mask = tf.sequence_mask([len(sent_tokens) for sent_tokens in self.tokens], constants.MAX_TOKENS)

		for sent_tokens, sentence_mask in zip(self.tokens, tf.unstack(seq_mask)):
			sentence_mask = tf.cast(sentence_mask, tf.float32)
			sentence_mask = tf.expand_dims(sentence_mask, 1)
			sentence_mask = sentence_mask * tf.transpose(sentence_mask)

			sentence_length = min(len(sent_tokens), constants.MAX_TOKENS)  # All observation fields must be of same length
			sentence_distances = np.zeros((constants.MAX_TOKENS, constants.MAX_TOKENS), dtype=np.float32)

			random_tree = self.generate_random_tree(sentence_length)
			for i in range(sentence_length):
				for j in range(i, sentence_length):
					i_j_distance = self.distance_between_pairs(random_tree, i, j)
					sentence_distances[i, j] = i_j_distance
					sentence_distances[j, i] = i_j_distance

			yield tf.constant(sentence_distances, dtype=tf.float32), sentence_mask


class RandomDepth(DependencyDepth):

	max_wordpieces = constants.MAX_WORDPIECES

	def __init__(self, conll_file, bert_tokenizer, lang='en'):
		super().__init__(conll_file, bert_tokenizer)

	def target_and_mask(self):
		"""Computes the depth of each word; returns them as a tensor.

		Returns:
		  target: A tensor of shape (num_examples, sentence_length,) of depths in the parse tree.
		  mask: A tensor of shape (number of examples, sentence_length) specifying which elements of the target
		  should be used during training.
		"""
		seq_mask = tf.sequence_mask([len(sent_tokens) for sent_tokens in self.tokens], constants.MAX_TOKENS)

		for sent_tokens, sentence_mask in zip(self.tokens, tf.unstack(seq_mask)):
			sentence_mask = tf.cast(sentence_mask, tf.float32)
			sentence_length = min(len(sent_tokens), constants.MAX_TOKENS)  # All observation fields must be of same length
			sentence_depths = np.zeros(constants.MAX_TOKENS, dtype=np.float32)

			random_tree = self.generate_random_tree(sentence_length)
			for i in range(sentence_length):
				sentence_depths[i] = self.get_ordering_index(random_tree, i)

			yield tf.constant(sentence_depths, dtype=tf.float32), sentence_mask
