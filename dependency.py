from collections import defaultdict, OrderedDict
import numpy as np
import tensorflow as tf

from unidecode import unidecode

import constants


class Dependency():

    def __init__(self, conll_file, bert_tokenizer):

        self.conllu_name = conll_file
        self.tokenizer = bert_tokenizer

        self.tokens = []
        self.lemmas = []
        self.pos = []
        self.relations = []
        self.roots = []
        self.features = []

        self.read_conllu(conll_file)

    @property
    def unlabeled_relations(self):
        return [{dep: parent for dep, parent in sent_relation} for sent_relation in self.relations]

    @property
    def punctuation_mask(self):
        return [[pos_tag != "PUNCT" for pos_tag in sentence_pos] for sentence_pos in self.pos]

    @property
    def unlabeled_unordered_relations(self):
        return [{frozenset((dep, parent)) for dep, parent in sent_relation
                 if (not sent_punctuation_mask[dep - 1]) and parent}
                for sent_relation, sent_punctuation_mask in zip(self.relations, self.punctuation_mask)]
    
    @property
    def word_count(self):
        return [len(sent_relation) for sent_relation in self.relations]

    @staticmethod
    def parse_features(feature_line):
        feats = dict()
        if feature_line == '_':
            return feats
        for feat in feature_line.split('|'):
            assert(len(feat.split('=')) == 2), 'Wrong formatting of features!'
            f_key, f_value = feat.split('=')
            feats[f_key] = f_value
        return feats
    
    def remove_indices(self, indices_to_rm):
        if self.tokens:
            self.tokens = [v for i, v in enumerate(self.tokens) if i not in indices_to_rm]
        if self.lemmas:
            self.lemmas = [v for i, v in enumerate(self.lemmas) if i not in indices_to_rm]
        if self.pos:
            self.pos = [v for i, v in enumerate(self.pos) if i not in indices_to_rm]
        if self.relations:
            self.relations = [v for i, v in enumerate(self.relations) if i not in indices_to_rm]
        if self.roots:
            self.roots = [v for i, v in enumerate(self.roots) if i not in indices_to_rm]
        if self.features:
            self.features = [v for i, v in enumerate(self.features) if i not in indices_to_rm]

    def read_conllu(self, conll_file_path):
        sentence_relations = []
        sentence_tokens = []
        sentence_lemmas = []
        sentence_pos = []
        sentence_features = []

        with open(conll_file_path, 'r') as in_conllu:
            sentid = 0
            for line in in_conllu:
                if line == '\n':
                    self.relations.append(sentence_relations)
                    sentence_relations = []
                    self.tokens.append(sentence_tokens)
                    sentence_tokens = []
                    self.lemmas.append(sentence_lemmas)
                    sentence_lemmas = []
                    self.pos.append(sentence_pos)
                    sentence_pos = []
                    self.features.append(sentence_features)
                    sentence_features = []

                    sentid += 1
                elif line.startswith('#'):
                    continue
                else:
                    fields = line.strip().split('\t')
                    if fields[constants.CONLLU_ID].isdigit():
                        head_id = int(fields[constants.CONLLU_HEAD])
                        dep_id = int(fields[constants.CONLLU_ID])
                        sentence_relations.append((dep_id, head_id))
                        sentence_features.append(self.parse_features(fields[constants.CONLLU_FEATS]))

                        if head_id == 0:
                            self.roots.append(int(fields[constants.CONLLU_ID]))

                        sentence_tokens.append(fields[constants.CONLLU_ORTH])
                        sentence_lemmas.append(fields[constants.CONLLU_LEMMA])
                        sentence_pos.append(fields[constants.CONLLU_POS])

    def get_bert_ids(self, wordpieces):
        """Token ids from Tokenizer vocab"""
        token_ids = self.tokenizer.convert_tokens_to_ids(wordpieces)
        input_ids = token_ids + [0] * (constants.MAX_WORDPIECES - len(wordpieces))
        return input_ids

    def training_examples(self):
        '''
        Joins wordpices of tokens, so that they correspond to the tokens in conllu file.
        :param wordpieces_all: lists of BPE pieces for each sentence
        :return:
            2-D tensor  [num valid sentences, max num wordpieces] bert wordpiece ids,
            2-D tensor [num valid sentences, max num wordpieces] wordpiece to word segment mappings
            1-D tensor [num valide sentenes] number of words in each sentence
        '''
        
        number_examples = len(self.tokens)
        wordpieces = []
        indices_to_rm = []
        for idx, sent_tokens in enumerate(self.tokens[:]):
            sent_wordpieces = ["[CLS]"] + self.tokenizer.tokenize((' '.join(sent_tokens)), add_special_tokens=False) + ["[SEP]"]
            wordpieces.append(sent_wordpieces)
            if len(sent_tokens) >= constants.MAX_TOKENS:
                print(f"Sentence {idx} too many tokens in file {self.conllu_name}, skipping.")
                indices_to_rm.append(idx)
                number_examples -= 1
            elif len(sent_wordpieces) >= constants.MAX_WORDPIECES:
                print(f"Sentence {idx} too many wordpieces in file {self.conllu_name}, skipping.")
                indices_to_rm.append(idx)
                number_examples -= 1
        
        segments = []
        max_segment = []
        bert_ids = []
        sent_idx = 0
        for sent_wordpieces, sent_tokens in zip(wordpieces, self.tokens):
            if sent_idx in indices_to_rm:
                sent_idx += 1
                continue
            
            sent_segments = np.zeros((constants.MAX_WORDPIECES,), dtype=np.int64) - 1
            segment_id = 0
            wordpiece_pointer = 1
            for token in sent_tokens:
                worpieces_per_token = len(self.tokenizer.tokenize(token, add_special_tokens=False))
                sent_segments[wordpiece_pointer:wordpiece_pointer+worpieces_per_token] = segment_id
                wordpiece_pointer += worpieces_per_token
                segment_id += 1
                    
            if wordpiece_pointer+1 != len(sent_wordpieces):
                print(f'Sentence {sent_idx} mismatch in number of tokens, skipped!')
                indices_to_rm.append(sent_idx)
            else:
                segments.append(tf.constant(sent_segments, dtype=tf.int64))
                bert_ids.append(tf.constant(self.get_bert_ids(sent_wordpieces), dtype=tf.int64))
                max_segment.append(segment_id)
            sent_idx += 1
    
        self.remove_indices(indices_to_rm)
        
        return bert_ids, segments, max_segment
        
    
class DependencyDistance(Dependency):
    
    def __init__(self, conll_file, bert_tokenizer):
        super().__init__(conll_file, bert_tokenizer)

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

        distances = []
        for dependency_tree, sentence_mask in zip(self.relations, tf.unstack(seq_mask)):
            sentence_length = min(len(dependency_tree), constants.MAX_TOKENS)  # All observation fields must be of same length
            sentence_distances = np.zeros((constants.MAX_TOKENS, constants.MAX_TOKENS), dtype=np.float32)
            for i in range(sentence_length):
                for j in range(i, sentence_length):
                    i_j_distance = self.distance_between_pairs(dependency_tree, i, j)
                    sentence_distances[i, j] = i_j_distance
                    sentence_distances[j, i] = i_j_distance
                    
            #distances.append(sentence_distances)
            yield sentence_distances, sentence_mask

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
    
    
class DependencyDepth(Dependency):
    
    def __init__(self, conll_file, bert_tokenizer):
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

        depths = []
        for dependency_tree, sentence_mask in zip(self.relations, tf.unstack(seq_mask)):
            sentence_length = min(len(dependency_tree), constants.MAX_TOKENS)  # All observation fields must be of same length
            sentence_depths = np.zeros(constants.MAX_TOKENS, dtype=np.float32)
            for i in range(sentence_length):
                sentence_depths[i] = self.get_ordering_index(dependency_tree, i)
            depths.append(sentence_depths)
        
            yield sentence_depths, sentence_mask

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

#TODO: feature distance, decide whether we want to examine this
# class FeatureDistance(Dependency):
#
#     def __init__(self, conll_file, bert_tokenizer):
#         super().__init__(conll_file, bert_tokenizer)
#
#     def target_and_mask(self):
#         """Computes the edit distances between morphological feuatures ; returns them as a tensor.
#         Masks tokens whithout any morphologicall features
#
#         Returns:
#           A tensor of shape (num exmaples, sentence_length, sentence_length) of distances
#           in the parse tree as specified by the observation annotation.
#         """
#         distances = []
#         masks = []
#
#         seq_mask = tf.cast(tf.sequence_mask([len(sent_tokens) for sent_tokens in self.tokens], constants.MAX_TOKENS),
#                        tf.float32)
#         seq_mask = tf.expand_dims(seq_mask, 1)
#         seq_mask = seq_mask * tf.transpose(seq_mask, perm=[0, 2, 1])
#
#         for sentence_features in self.features:
#             sentence_length = len(sentence_features)  # All observation fields must be of same length
#             sentence_distances = np.zeros((constants.MAX_TOKENS, constants.MAX_TOKENS), dtype=np.float32)
#             sentence_mask = np.zeros((constants.MAX_TOKENS, constants.MAX_TOKENS), dtype=np.float32)
#             for i in range(sentence_length):
#                 for j in range(i, sentence_length):
#                     i_j_distance = self.distance_between_pairs(sentence_features, i, j)
#                     if i_j_distance:
#                         sentence_distances[i, j] = i_j_distance
#                         sentence_distances[j, i] = i_j_distance
#                         sentence_mask[i, j] = 1.
#                         sentence_mask[j, i] = 1.
#
#             distances.append(sentence_distances)
#             masks.append(sentence_mask)
#         return tf.cast(tf.stack(distances), dtype=tf.float32), masks * seq_mask
#
#     @staticmethod
#     def distance_between_pairs(sentence_features, i, j):
#         if not sentence_features[i] or not sentence_features[j]:
#             return None
#
#         i_f_keys = set(sentence_features[i].keys)
#         j_f_keys = set(sentence_features[j].keys)
#
#         shared_feats = 0
#         for f_key in i_f_keys.intersection(j_f_keys):
#             if sentence_features[i][f_key] == sentence_features[j][f_key]:
#                 shared_feats += 1
#
#         return 1. - shared_feats / len(i_f_keys.union(j_f_keys))