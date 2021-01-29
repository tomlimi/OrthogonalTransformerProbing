import numpy as np
import tensorflow as tf

import constants
from data_support.conll_wrapper import ConllWrapper


class ShuffledDistance(ConllWrapper):

	def __init__(self, conll_file, bert_tokenizer, lang='en'):
		super().__init__(conll_file, bert_tokenizer)

	def target_and_mask(self):
		"""Computes the distances between all pairs of words; returns them as a tensor.

		Returns:
		  target: A tensor of shape (num_examples, sentence_length, sentence_length) of positional distances.
		  mask: A tensor of shape (number of examples, sentence_length, sentence_length) specifying which elements of
		  the target should be used during training.
		"""
		seq_mask = tf.sequence_mask([len(sent_tokens) for sent_tokens in self.tokens], constants.MAX_TOKENS)

		for sent_tokens, sent_shuf, sentence_mask in zip(self.tokens, self.shuffled, tf.unstack(seq_mask)):
			sentence_mask = tf.cast(sentence_mask, tf.float32)
			sentence_mask = tf.expand_dims(sentence_mask, 1)
			sentence_mask = sentence_mask * tf.transpose(sentence_mask)

			sentence_length = min(len(sent_tokens), constants.MAX_TOKENS)  # All observation fields must be of same length
			sentence_distances = np.zeros((constants.MAX_TOKENS, constants.MAX_TOKENS), dtype=np.float32)
			for i, i_idx in enumerate(sent_shuf):
				for j, j_idx in zip(range(i, sentence_length), sent_shuf[i:]):
					i_j_distance = j - i
					sentence_distances[i_idx, j_idx] = i_j_distance
					sentence_distances[j_idx, i_idx] = i_j_distance
			# sentence_distances = sentence_distances[sent_shuf,:]
			# sentence_distances = sentence_distances[:,sent_shuf]

			#sentence_mask = tf.linalg.set_diag(sentence_mask, tf.repeat(0., constants.MAX_TOKENS))

			yield tf.constant(sentence_distances, dtype=tf.float32), sentence_mask

	def training_examples(self, shuffle=True):
		super().training_examples(shuffle)


class ShuffledDepth(ConllWrapper):

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

		for sent_tokens, sent_shuf, sentence_mask in zip(self.tokens, self.shuffled, tf.unstack(seq_mask)):
			sentence_mask = tf.cast(sentence_mask, tf.float32)
			sentence_length = min(len(sent_tokens), constants.MAX_TOKENS)  # All observation fields must be of same length
			sentence_depths = np.zeros(constants.MAX_TOKENS, dtype=np.float32)
			for i, i_idx in enumerate(sent_shuf):
				sentence_depths[i_idx] = float(i)

			yield tf.constant(sentence_depths, dtype=tf.float32), sentence_mask
			
	def training_examples(self, shuffle=True):
		super().training_examples(shuffle)
