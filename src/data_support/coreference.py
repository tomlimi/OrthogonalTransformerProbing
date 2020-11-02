import numpy as np
import tensorflow as tf
from itertools import combinations, chain
from collections import defaultdict
import networkx as nx
import re

import constants
from data_support.conll_wrapper import ConllWrapper


MAX_COREF_DISTANCE = 10

# look only at the coreferents that are nouns or pronouns
COREF_POS_LIST = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'WP', 'WP$', 'WRB']


class CoreferenceDistance(ConllWrapper):

	max_wordpieces = constants.MAX_WORDPIECES

	def __init__(self, conll_file, bert_tokenizer, lang='en'):
		super().__init__(conll_file, bert_tokenizer)

	@staticmethod
	def process_coreference(coref, coreference_line):

		curr_coref = coref[:]
		next_coref = coref[:]
		if coreference_line == '_':
			return curr_coref, next_coref


		for coref_entity in coreference_line.split('|'):
			if coref_entity.startswith('(') and coref_entity.endswith(')'):
				coref_idx = int(coref_entity[1:-1])
				# self.coref_counter[coref_entity] += 1
				curr_coref.append(coref_idx)
			elif coref_entity.startswith('('):
				coref_idx = int(coref_entity[1:])
				curr_coref.append(coref_idx)
				next_coref.append(coref_idx)
			elif coref_entity.endswith(')'):

				coref_idx = int(coref_entity[:-1])
				try:
					next_coref.remove(coref_idx)
				except ValueError:
					print("Coreference ended before begin!")

		return curr_coref, next_coref

	def read_conllu(self, conll_file_path, lang='en'):
		sentence_tokens = []
		sentence_coreference = []
		sentence_pos = []

		with open(conll_file_path, 'r') as in_conllu:
			docid = 0
			sentindoc = 0

			curr_coref = []
			for line in in_conllu:
				if line == '\n':

					self.tokens.append(sentence_tokens)
					sentence_tokens = []

					self.coreferences.append(sentence_coreference)
					sentence_coreference = []

					self.pos.append(sentence_pos)
					sentence_pos = []

					docid += 1
					sentindoc = 0

				elif line.startswith('#'):
					continue
				else:
					fields = line.strip().split('\t')
					if fields[constants.CONLLU_ID].isdigit():

						sentids = re.findall(r"-s[0-9]+-", fields[constants.CONLL_NODE])
						if len(sentids) != 1:
							raise ValueError("Wrong format of node name in conll file!")
						sentid = int(sentids[0][2:-1])

						# One train example will consist of two sentences at mosts.
						if sentid > sentindoc:
							sentindoc = sentid
							if sentindoc != 1 and sentindoc % 2 == 1:
								self.tokens.append(sentence_tokens)
								sentence_tokens = []

								self.coreferences.append(sentence_coreference)
								sentence_coreference = []

								self.pos.append(sentence_pos)
								sentence_pos = []

						sentence_tokens.append(fields[constants.CONLLU_ORTH])
						coref, curr_coref = self.process_coreference(curr_coref, fields[constants.CONLL_COREF])
						sentence_coreference.append(coref)

						sentence_pos.append(fields[constants.CONLL_POS])

	@staticmethod
	def coreferents_distances(coreference_list):
		coreference_graph = nx.Graph()
		coreference_graph.add_nodes_from(chain(*coreference_list))

		for coreferents in coreference_list:
			if len(coreferents) >= 2:
				for coref_a, coref_b in combinations(coreferents, 2):
					coreference_graph.add_edge(coref_a, coref_b)

		#dict of dict with pairwise distances between coreferences, if it exists
		return dict(nx.all_pairs_shortest_path_length(coreference_graph))

	def target_and_mask(self):
		"""Computes the distances between all pairs of words; returns them as a tensor.

		Returns:
		  target: A tensor of shape (num_examples, sentence_length, sentence_length) of distances
		  in the parse tree.
		  mask: A tensor of shape (number of examples, sentence_length, sentence_length) specifying which elements of
		  the target should be used during training.
		"""
		seq_mask = tf.cast(tf.sequence_mask([len(sent_tokens) for sent_tokens in self.tokens], constants.MAX_TOKENS),
		                   tf.float32)
		seq_mask = tf.expand_dims(seq_mask, 1)
		seq_mask = seq_mask * tf.transpose(seq_mask, perm=[0, 2, 1])

		for coreferents_list, sentence_pos, sentence_seq_mask in zip(self.coreferences, self.pos, tf.unstack(seq_mask)):
			sentence_length = min(len(coreferents_list),
			                      constants.MAX_TOKENS)  # All observation fields must be of same length
			sentence_distances = np.zeros((constants.MAX_TOKENS, constants.MAX_TOKENS), dtype=np.float32)

			sentence_mask = np.zeros((constants.MAX_TOKENS, constants.MAX_TOKENS), dtype=np.float32)

			coreferents_distances = self.coreferents_distances(coreferents_list)
			for i in range(sentence_length):
				for j in range(i, sentence_length):
					if sentence_pos[i] in COREF_POS_LIST and sentence_pos[j] in COREF_POS_LIST \
							and coreferents_list[i] and coreferents_list[j]:

						i_j_distance = self.distance_between_pairs(coreferents_distances, coreferents_list, i, j)
						sentence_distances[i, j] = i_j_distance
						sentence_distances[j, i] = i_j_distance
						sentence_mask[i,j] = 1.
						sentence_mask[j,i] = 1.

			# sentence_mask = tf.linalg.set_diag(sentence_mask, tf.repeat(0., constants.MAX_TOKENS))

			yield tf.constant(sentence_distances, dtype=tf.float32), tf.constant(sentence_mask, tf.float32) * sentence_seq_mask

	@staticmethod
	def distance_between_pairs(coreferents_distances, coreferents_list, i, j):
		'''Computes path distance between a pair of words

		Args:
		  coreferents_distances: dictionary of dictionaries of distances between coreferents
		  coreferents_list : list of coreferents per token
		  i: one of the two words to compute the distance between.
		  j: one of the two words to compute the distance between.

		Returns:
		  The integer distance d_path(i,j)
		'''
		if i == j:
			return 0
		corefs_i = coreferents_list[i]
		corefs_j = coreferents_list[j]

		# if set(corefs_i) & set(corefs_j):
		# 	return 0
		# else:
		# 	return MAX_COREF_DISTANCE
		#
		# if not corefs_i or not corefs_j:
		# 	return MAX_COREF_DISTANCE

		coref_distance = MAX_COREF_DISTANCE
		for coref_i in corefs_i:
			for coref_j in corefs_j:
				if coref_j in coreferents_distances[coref_i]:
					coref_distance = min(coref_distance, coreferents_distances[coref_i][coref_j] + 1)
					# + 1 here because distance between corferents should be also positive to distinguish
					# from the same word
		return coref_distance
