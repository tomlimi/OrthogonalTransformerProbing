import numpy as np
import tensorflow as tf

import constants
from data_support.conll_wrapper import ConllWrapper


class DependencyDistance(ConllWrapper):

    def __init__(self, conll_file, bert_tokenizer, lang=None):
        super().__init__(conll_file, bert_tokenizer, resize_examples=False)

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

        for dependency_tree, sentence_mask in zip(self.relations, tf.unstack(seq_mask)):
            sentence_length = min(len(dependency_tree), constants.MAX_TOKENS)  # All observation fields must be of same length
            sentence_distances = np.zeros((constants.MAX_TOKENS, constants.MAX_TOKENS), dtype=np.float32)
            for i in range(sentence_length):
                for j in range(i, sentence_length):
                    i_j_distance = self.distance_between_pairs(dependency_tree, i, j)
                    sentence_distances[i, j] = i_j_distance
                    sentence_distances[j, i] = i_j_distance

            #sentence_mask = tf.linalg.set_diag(sentence_mask, tf.repeat(0., constants.MAX_TOKENS))

            yield tf.constant(sentence_distances, dtype=tf.float32), sentence_mask

    @staticmethod
    def distance_between_pairs(dependency_tree, i, j):
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

        head_indices = []
        for dep, head in dependency_tree:
            head_indices.append(int(head))
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


class DependencyDepth(ConllWrapper):

    def __init__(self, conll_file, bert_tokenizer, lang=None):
        super().__init__(conll_file, bert_tokenizer, resize_examples=False)

    def target_and_mask(self):
        """Computes the depth of each word; returns them as a tensor.

        Returns:
          target: A tensor of shape (num_examples, sentence_length,) of depths in the parse tree.
          mask: A tensor of shape (number of examples, sentence_length) specifying which elements of the target
          should be used during training.
        """
        seq_mask = tf.cast(tf.sequence_mask([len(sent_tokens) for sent_tokens in self.tokens], constants.MAX_TOKENS),
                           tf.float32)

        for dependency_tree, sentence_mask in zip(self.relations, tf.unstack(seq_mask)):
            sentence_length = min(len(dependency_tree), constants.MAX_TOKENS)  # All observation fields must be of same length
            sentence_depths = np.zeros(constants.MAX_TOKENS, dtype=np.float32)
            for i in range(sentence_length):
                sentence_depths[i] = self.get_ordering_index(dependency_tree, i)

            yield tf.constant(sentence_depths, dtype=tf.float32), sentence_mask

    @staticmethod
    def get_ordering_index(dependency_tree, i):
        '''Computes tree depth for a single word in a sentence

        Args:
          dependency_tree: list of tuples (dependent, head) sorted by dependent indicies.
          i: the word in the sentence to compute the depth of

        Returns:
          The integer depth in the tree of word i
        '''

        head_indices = []
        for dep, head in dependency_tree:
            head_indices.append(int(head))
        length = 0
        i_head = i+1
        while True:
            i_head = head_indices[i_head - 1]
            if i_head != 0:
                length += 1
            else:
                return length
