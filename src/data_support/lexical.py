import numpy as np
import tensorflow as tf
from functools import lru_cache
from nltk.corpus import wordnet as wn

from data_support.conll_wrapper import ConllWrapper
import constants


class LexicalDistance(ConllWrapper):

    max_wordpieces = constants.MAX_WORDPIECES

    def __init__(self, conll_file, bert_tokenizer, lang='en'):
        super().__init__(conll_file, bert_tokenizer)
        if lang not in constants.lang2iso:
            raise ValueError(f'Language {lang} is not supported by Open Multilingual Wordnet')
        self.iso_lang = constants.lang2iso[lang]

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

        for sentence_pos, sentence_lemmas, sentence_seq_mask in zip(self.pos, self.lemmas, tf.unstack(seq_mask)):
            sentence_length = min(len(sentence_pos), constants.MAX_TOKENS)  # All observation fields must be of same length
            sentence_distances = np.zeros((constants.MAX_TOKENS, constants.MAX_TOKENS), dtype=np.float32)
            sentence_mask = np.zeros((constants.MAX_TOKENS, constants.MAX_TOKENS), dtype=np.float32)
            for i in range(sentence_length):
                # simialarity of word with itself is masked, because it is to simple exampl
                for j in range(i, sentence_length):
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

            yield tf.constant(sentence_distances, tf.float32), tf.constant(sentence_mask, tf.float32) * sentence_seq_mask
            
        self.distance_between_pairs.cache_clear()

    @lru_cache(maxsize=1024)
    def distance_between_pairs(self, lemma_i, lemma_j, pos_i, pos_j):
        '''Computes path distance between a pair of words

        Args:
          lemma_i: i-th word lemma.
          lemma_j: j-th word lemma.
          pos_i: i-th word part of speech tag.
          pos_j: j-th word part of speech tag.

        Returns:
          The minimal distance in the WordNet lexical tree d_path(i,j)
        '''

        if pos_i not in constants.pos2wnpos or pos_j not in constants.pos2wnpos:
            return None
        if not wn.synsets(lemma_i, pos=constants.pos2wnpos[pos_i], lang=self.iso_lang) or \
                not wn.synsets(lemma_j, pos=constants.pos2wnpos[pos_j], lang=self.iso_lang):
            return None

        max_similarity = 0.
        # TODO: consider language, maybe use other type of similatity
        for i_synset in wn.synsets(lemma_i, pos=constants.pos2wnpos[pos_i], lang=self.iso_lang):
            for j_synset in wn.synsets(lemma_j, pos=constants.pos2wnpos[pos_j], lang=self.iso_lang):
                pair_sim = wn.path_similarity(i_synset, j_synset)
                if pair_sim and pair_sim > max_similarity:
                    max_similarity = pair_sim

        if max_similarity == 0.:
            return None

        return 1./max_similarity


class LexicalDepth(ConllWrapper):

    max_wordpieces = constants.MAX_WORDPIECES

    def __init__(self, conll_file, bert_tokenizer, lang='en'):
        super().__init__(conll_file, bert_tokenizer)
        if lang not in constants.lang2iso:
            raise ValueError(f'Language {lang} is not supported by Open Multilingual Wordnet')
        self.iso_lang = constants.lang2iso[lang]

    def target_and_mask(self):
        """Computes the depth of each word; returns them as a tensor.

        Returns:
          target: A tensor of shape (number of examples, sentence_length) of depths
          in the parse tree as specified by the observation annotation.
          mask: A tensor of shape (number of examples, sentence_length) specifying which elements of the target
          should be used during training.
        """

        seq_mask = tf.cast(tf.sequence_mask([len(sent_tokens) for sent_tokens in self.tokens], constants.MAX_TOKENS),
                           tf.float32)

        for sentence_pos, sentence_lemmas, sentence_seq_mask in zip(self.pos, self.lemmas, tf.unstack(seq_mask)):
            sentence_length = min(len(sentence_pos), constants.MAX_TOKENS) # All observation fields must be of same length
            sentence_depths = np.zeros(constants.MAX_TOKENS, dtype=np.float32)
            sentence_mask = np.zeros(constants.MAX_TOKENS, dtype=np.float32)
            for i in range(sentence_length):
                i_depth = self.get_ordering_index(sentence_lemmas[i], sentence_pos[i])
                if i_depth is not None:

                    sentence_depths[i] = i_depth
                    sentence_mask[i] = 1.

            yield tf.constant(sentence_depths, tf.float32), tf.constant(sentence_mask, tf.float32) * sentence_seq_mask

        self.get_ordering_index.cache_clear()

    @lru_cache(maxsize=1024)
    def get_ordering_index(self, lemma, pos):
        '''Computes tree depth for a single word in a sentence

        Args:
          dependency_tree: list of tuples (dependent, head) sorted by dependent indicies.
          i: the word in the sentence to compute the depth of

        Returns:
          The minimal integer depth in the WordNet lexical tree of word i
        '''

        if pos not in constants.pos2wnpos:
            return None

        if not wn.synsets(lemma, pos=constants.pos2wnpos[pos], lang=self.iso_lang):
            return None

        min_depth = np.inf
        for synset in wn.synsets(lemma, pos=constants.pos2wnpos[pos], lang=self.iso_lang):
            if synset.min_depth() < min_depth:
                min_depth = synset.min_depth()

        return min_depth
