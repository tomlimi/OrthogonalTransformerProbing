import numpy as np
import tensorflow as tf
from functools import lru_cache
from nltk.corpus import wordnet as wn

from dependency import Dependency
import constants


class LexicalDistance(Dependency):

    def __init__(self, conll_file, bert_tokenizer):
        super().__init__(conll_file, bert_tokenizer)

    def target_and_mask(self):
        """Computes the distances between all pairs of words; returns them as a tensor.

        Returns:
          target: A tensor of shape (number of examples, sentence_length, sentence_length) of distances
          in the parse tree as specified by the observation annotation.
          mask: A tensor of shape (number of examples, sentence_length, sentence_length) specifying which elements of
          the target should be used during training.
        """
        seq_mask = tf.cast(tf.sequence_mask([len(sent_tokens) for sent_tokens in self.tokens], constants.MAX_TOKENS),
                           tf.float32)
        seq_mask = tf.expand_dims(seq_mask, 1)
        seq_mask = seq_mask * tf.transpose(seq_mask, perm=[0, 2, 1])

        distances = []
        masks = []

        for sentence_pos, sentence_lemmas in zip(self.pos, self.lemmas):
            sentence_length = min(len(sentence_pos), constants.MAX_TOKENS)  # All observation fields must be of same length
            sentence_distances = np.zeros((constants.MAX_TOKENS, constants.MAX_TOKENS), dtype=np.float32)
            sentence_mask = np.zeros((constants.MAX_TOKENS, constants.MAX_TOKENS), dtype=np.float32)
            for i in range(sentence_length):
                # simialarity of word with itself is masked, because it is to simple exampl
                for j in range(i+1, sentence_length):
                    if sentence_lemmas[i] == sentence_lemmas[j] and sentence_pos[i] == sentence_pos[j]:
                        i_j_distance = 0
                    else:
                        i_j_distance = self.distance_between_pairs(sentence_lemmas[i], sentence_lemmas[j],
                                                               sentence_pos[i], sentence_pos[j])
                    if i_j_distance is not None:
                        sentence_distances[i, j] = i_j_distance
                        sentence_distances[j, i] = i_j_distance
                        sentence_mask[i, j] = 1.
                        sentence_mask[j, i] = 1.

            distances.append(sentence_distances)
            masks.append(sentence_mask)
        self.distance_between_pairs.cache_clear()
        return tf.cast(tf.stack(distances), dtype=tf.float32), masks * seq_mask

    @staticmethod
    @lru_cache(maxsize=2 ** 14)
    def distance_between_pairs(lemma_i, lemma_j, pos_i, pos_j):
        '''Computes path distance between a pair of words

        Args:
          lemma_i: i-th word lemma.
          lemma_j: j-th word lemma.
          pos_i: i-th word part of speech tag.
          pos_j: j-th word part of speech tag.

        Returns:
          The distance in the WordNet lexical tree d_path(i,j)
        '''

        if pos_i not in constants.pos2wnpos or pos_j not in constants.pos2wnpos:
            return None
        if not wn.synsets(lemma_i, pos=constants.pos2wnpos[pos_i]) or not wn.synsets(lemma_j, pos=constants.pos2wnpos[pos_j]):
            return None

        similarity = 0.
        # TODO: consider language, maybe use other type of similatity
        for i_synset in wn.synsets(lemma_i, pos=constants.pos2wnpos[pos_i]):
            for j_synset in wn.synsets(lemma_j, pos=constants.pos2wnpos[pos_j]):
                pair_sim = wn.path_similarity(i_synset, j_synset)
                if pair_sim and pair_sim > similarity:
                    similarity = pair_sim

        if similarity == 0.:
            return None

        return 1./similarity


class LexicalDepth(Dependency):

    def __init__(self, conll_file, bert_tokenizer):
        super().__init__(conll_file, bert_tokenizer)

    def target_and_mask(self):
        """Computes the depth of each word; returns them as a tensor.

        Returns:
          target: A tensor of shape (number of examples, sentence_length) of depths
          in the parse tree as specified by the observation annotation.
          mask: A tensor of shape (number of examples, sentence_length) specifying which elements of the target
          should be used during training.
        """

        raise NotImplementedError

        # seq_mask = tf.cast(tf.sequence_mask([len(sent_tokens) for sent_tokens in self.tokens], constants.MAX_TOKENS),
        #                    tf.float32)
        #
        # depths = []
        # for dependency_tree in self.relations:
        #     sentence_length = len(dependency_tree)  # All observation fields must be of same length
        #     sentence_depths = np.zeros(constants.MAX_TOKENS, dtype=np.float32)
        #     for i in range(sentence_length):
        #         sentence_depths[i] = self.get_ordering_index(dependency_tree, i)
        #     depths.append(sentence_depths)
        #
        # return tf.cast(tf.stack(depths), dtype=tf.float32), seq_mask
