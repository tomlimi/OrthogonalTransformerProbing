import csv
from collections import defaultdict
from itertools import chain

import tensorflow as tf
import numpy as np

from data_support.conll_wrapper import ConllWrapper
import constants

UDER = {'en': '../resources/UDer-1.0-en-CatVar.tsv'}


class Derivation(ConllWrapper):
	def __init__(self, conll_file, bert_tokenizer, lang):
		super().__init__(conll_file, bert_tokenizer)
		self.lemma_pos2tree_node = dict()
		self.trees_nodes = list()
		self.nodes_parents = list()

		self.read_derinet(lang)

	def read_derinet(self, lang):

		# First iteration to check wich records are proper trees (with two or more nodes in conll).
		lemma_pos_vocabulary = set((lemma, pos) for lemma, pos in
		                           zip(chain.from_iterable(self.lemmas), chain.from_iterable(self.pos)))
		tree_node_counter = defaultdict(int)
		dn_rows = list()
		with open(UDER[lang], 'r') as dn_file:
			for dn_row in csv.reader(dn_file, delimiter='\t'):
				if not dn_row:
					continue
				tree_idx, node_idx = map(int, dn_row[0].split('.'))
				lemma_pos = tuple(dn_row[2:4])
				dn_rows.append(dn_row)
				if lemma_pos in lemma_pos_vocabulary:
					tree_node_counter[tree_idx] += 1

		dn_rows = [dn_row for dn_row in dn_rows if tree_node_counter[int(dn_row[0].split('.')[0])] > 1]

		new_tree_idx = -1
		for dn_row in dn_rows:
			tree_idx, node_idx = map(int, dn_row[0].split('.'))
			if not node_idx:
				new_tree_idx += 1
				self.trees_nodes.insert(new_tree_idx, [])
				self.nodes_parents.insert(new_tree_idx, [])

			lemma_pos = tuple(dn_row[2:4])
			if lemma_pos not in lemma_pos_vocabulary:
				lemma_pos = None

			if dn_row[6]:
				parent_tree_idx, parent_node_idx = map(int, dn_row[6].split('.'))
				assert parent_tree_idx == tree_idx
				parent_node_idx += 1  # to be compatible with dependency tree annontations
			else:
				parent_node_idx = 0

			self.lemma_pos2tree_node[lemma_pos] = (new_tree_idx, node_idx)
			self.trees_nodes[new_tree_idx].append(lemma_pos)
			self.nodes_parents[new_tree_idx].append(parent_node_idx)


class DerivationDistance(Derivation):

	def __init__(self, conll_file, bert_tokenizer, lang='en'):
		super().__init__(conll_file, bert_tokenizer, lang=lang)


	def target_and_mask(self):
		"""Computes the distances between all pairs of words; returns them as a tensor.

		Returns:
		  target: A tensor of shape (num_examples, sentence_length, sentence_length) of distances
		  in the parse tree.
		  mask: A tensor of shape (number of examples, sentence_length, sentence_length) specifying which elements of
		  the target should be used during training.
		"""

		seq_mask = tf.cast(tf.sequence_mask(
			[len(sent_tokens) for sent_tokens in self.tokens], constants.MAX_TOKENS), tf.float32)

		seq_mask = tf.expand_dims(seq_mask, 1)
		seq_mask = seq_mask * tf.transpose(seq_mask, perm=[0, 2, 1])

		for lemma_pos, head_indices, sentence_seq_mask in zip(self.trees_nodes, self.nodes_parents, tf.unstack(seq_mask)):
			tree_size = min(len(head_indices), constants.MAX_TOKENS)  # All observation fields must be of same length
			sentence_distances = np.zeros((constants.MAX_TOKENS, constants.MAX_TOKENS), dtype=np.float32)
			sentence_mask = np.zeros((constants.MAX_TOKENS, constants.MAX_TOKENS), dtype=np.float32)
			for i in range(tree_size):
				for j in range(i, tree_size):
					if lemma_pos[i] and lemma_pos[j]:
						i_j_distance = self.distance_between_pairs(head_indices, i, j)
						sentence_distances[i, j] = i_j_distance
						sentence_distances[j, i] = i_j_distance

						sentence_mask[i, j] = 1.
						sentence_mask[j, i] = 1.

			# sentence_mask = tf.linalg.set_diag(sentence_mask, tf.repeat(0., constants.MAX_TOKENS))

			yield tf.constant(sentence_distances, dtype=tf.float32), tf.constant(sentence_mask, tf.float32) * sentence_seq_mask

	@staticmethod
	def distance_between_pairs(head_indices, i, j):
		'''Computes path distance between a pair of words

		Args:
		  dependency_tree: list of tuples (dependent, head) sorted by dependent indicies.
		  i: one of the two words to compute the distance between.
		  j: one of the two words to compute the distance between.

		Returns:
		  The integer distance d_path(i,j)
		'''
		if i == j:
			return 0

		i_path = [i + 1]
		j_path = [j + 1]
		i_head = i + 1
		j_head = j + 1
		while True:
			if not (i_head == 0 and (i_path == [i + 1] or i_path[-1] == 0)):
				i_head = head_indices[i_head - 1]
				i_path.append(i_head)
			if not (j_head == 0 and (j_path == [j + 1] or j_path[-1] == 0)):
				j_head = head_indices[j_head - 1]
				j_path.append(j_head)
			if i_head in j_path:
				j_path_length = j_path.index(i_head)
				i_path_length = len(i_path) - 1
				break
			elif j_head in i_path:
				i_path_length = i_path.index(j_head)
				j_path_length = len(j_path) - 1
				break
			elif i_head == j_head:
				i_path_length = len(i_path) - 1
				j_path_length = len(j_path) - 1
				break
		total_length = j_path_length + i_path_length
		return total_length


class DerivationDepth(Derivation):

	def __init__(self, conll_file, bert_tokenizer, lang='en'):
		super().__init__(conll_file, bert_tokenizer, lang=lang)

	def target_and_mask(self):
		"""Computes the depth of each word; returns them as a tensor.

		Returns:
		  target: A tensor of shape (num_examples, sentence_length,) of depths in the parse tree.
		  mask: A tensor of shape (number of examples, sentence_length) specifying which elements of the target
		  should be used during training.
		"""

		seq_mask = tf.cast(tf.sequence_mask(
			[len(sent_tokens) for sent_tokens in self.tokens], constants.MAX_TOKENS), tf.float32)

		for lemma_pos, head_indices, sentence_seq_mask in zip(self.trees_nodes, self.nodes_parents, tf.unstack(seq_mask)):
			tree_size = min(len(head_indices), constants.MAX_TOKENS)  # All observation fields must be of same length
			sentence_depths = np.zeros(constants.MAX_TOKENS, dtype=np.float32)
			sentence_mask = np.ones(constants.MAX_TOKENS, dtype=np.float32)
			for i in range(tree_size):
				if lemma_pos[i]:
					sentence_depths[i] = self.get_ordering_index(head_indices, i)
					sentence_mask[i] = 1.

			yield tf.constant(sentence_depths, dtype=tf.float32), tf.constant(sentence_mask, tf.float32) * sentence_seq_mask

	@staticmethod
	def get_ordering_index(head_indices, i):
		'''Computes tree depth for a single word in a sentence

		Args:
		  dependency_tree: list of tuples (dependent, head) sorted by dependent indicies.
		  i: the word in the sentence to compute the depth of

		Returns:
		  The integer depth in the tree of word i
		'''

		length = 0
		i_head = i + 1
		while True:
			i_head = head_indices[i_head - 1]
			if i_head != 0:
				length += 1
			else:
				return length


