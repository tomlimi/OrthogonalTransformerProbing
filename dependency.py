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
        self.relations = []
        self.roots = []
        self.punctuation_mask = []

        self.read_conllu(conll_file)

    @property
    def unlabeled_relations(self):
        return [{dep: parent for dep, parent in sent_relation} for sent_relation in self.relations]
    
    @property
    def unlabeled_unordered_relations(self):
        return [{frozenset((dep, parent)) for dep, parent in sent_relation
                 if not sent_punctuation_mask[dep - 1]}
                for sent_relation, sent_punctuation_mask in zip(self.relations, self.punctuation_mask)]
    
    @property
    def word_count(self):
        return [len(sent_relation) for sent_relation in self.relations]
    
    def remove_indices(self, indices_to_rm):
        if self.tokens:
            self.tokens = [v for i, v in enumerate(self.tokens) if i not in indices_to_rm]
        if self.relations:
            self.relations = [v for i, v in enumerate(self.relations) if i not in indices_to_rm]
        if self.roots:
            self.roots = [v for i, v in enumerate(self.roots) if i not in indices_to_rm]
        if self.punctuation_mask:
            self.punctuation_mask = [v for i, v in enumerate(self.punctuation_mask) if i not in indices_to_rm]

    def read_conllu(self, conll_file_path):
        sentence_relations = []
        sentence_tokens = []
        sentence_punctuation_mask = []

        with open(conll_file_path, 'r') as in_conllu:
            sentid = 0
            for line in in_conllu:
                if line == '\n':
                    self.relations.append(sentence_relations)
                    sentence_relations = []
                    self.tokens.append(sentence_tokens)
                    sentence_tokens = []
                    self.punctuation_mask.append(sentence_punctuation_mask)
                    sentence_punctuation_mask = []
                    sentid += 1
                elif line.startswith('#'):
                    continue
                else:
                    fields = line.strip().split('\t')
                    if fields[constants.CONLLU_ID].isdigit():
                        head_id = int(fields[constants.CONLLU_HEAD])
                        dep_id = int(fields[constants.CONLLU_ID])
                        sentence_relations.append((dep_id, head_id))

                        if head_id == 0:
                            self.roots.append(int(fields[constants.CONLLU_ID]))

                        sentence_tokens.append(fields[constants.CONLLU_ORTH])
                        
                        sentence_punctuation_mask.append(fields[constants.CONLLU_POS] == 'PUNCT')

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
            sent_wordpieces = ["[CLS]"] + self.tokenizer.tokenize((' '.join(sent_tokens))) + ["[SEP]"]
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
            curr_token = ''
            for wp_id, wp in enumerate(sent_wordpieces):
                if wp in ('[CLS]', '[SEP]'):
                    continue
                    
                sent_segments[wp_id] = segment_id
                if wp.startswith('##'):
                    curr_token += wp[2:]
                else:
                    curr_token += wp
                if unidecode(curr_token).lower() == unidecode(sent_tokens[segment_id]).lower():
                    segment_id += 1
                    curr_token = ''
                    
            if segment_id != len(sent_tokens):
                print(f'Sentence {sent_idx} mismatch in number of tokens, skipped!')
                indices_to_rm.append(sent_idx)
            else:
                segments.append(tf.constant(sent_segments, dtype=tf.int64))
                bert_ids.append(tf.constant(self.get_bert_ids(sent_wordpieces), dtype=tf.int64))
                max_segment.append(segment_id)
            sent_idx += 1
    
        self.remove_indices(indices_to_rm)
        
        return tf.stack(bert_ids), tf.stack(segments), tf.constant(max_segment)
        
    
class DependencyDistance(Dependency):
    
    def __init__(self, conll_file, bert_tokenizer):
        super().__init__(conll_file, bert_tokenizer)

    def target_tensor(self):
        """Computes the distances between all pairs of words; returns them as a torch tensor.

        Args:
          observation: a single Observation class for a sentence:
        Returns:
          A tensor of shape (sentence_length, sentence_length) of distances
          in the parse tree as specified by the observation annotation.
        """
        distances = []
        for dependency_tree in self.relations:
            sentence_length = len(dependency_tree)  # All observation fields must be of same length
            sentence_distances = np.zeros((constants.MAX_TOKENS, constants.MAX_TOKENS), dtype=np.float32)
            for i in range(sentence_length):
                for j in range(i, sentence_length):
                    i_j_distance = DependencyDistance.distance_between_pairs(dependency_tree, i, j)
                    sentence_distances[i, j] = i_j_distance
                    sentence_distances[j, i] = i_j_distance
                    
            distances.append(sentence_distances)
        return tf.cast(tf.stack(distances), dtype=tf.float32)

    @staticmethod
    def distance_between_pairs(dependency_tree, i, j, head_indices=None):
        '''Computes path distance between a pair of words

        Args:
          dependency_tree: list of tuples (dependent, head) sorted by dependent indicies.
          i: one of the two words to compute the distance between.
          j: one of the two words to compute the distance between.
          head_indices: the head indices (according to a dependency parse) of all
              words, or None, if observation != None.

        Returns:
          The integer distance d_path(i,j)
        '''
        if i == j:
            return 0
        if dependency_tree:
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
        
    def target_tensor(self):
        """Computes the depth of each word; returns them as a torch tensor.

        Args:
          observation: a single Observation class for a sentence:
        Returns:
          A torch tensor of shape (sentence_length,) of depths
          in the parse tree as specified by the observation annotation.
        """
        depths = []
        for dependency_tree in self.relations:
            sentence_length = len(dependency_tree) #All observation fields must be of same length
            sentence_depths = np.zeros(constants.MAX_TOKENS, dtype=np.float32)
            for i in range(sentence_length):
                sentence_depths[i] = DependencyDepth.get_ordering_index(dependency_tree, i)
            depths.append(sentence_depths)
        
        return tf.cast(tf.stack(depths), dtype=tf.float32)

    @staticmethod
    def get_ordering_index(dependency_tree, i, head_indices=None):
        '''Computes tree depth for a single word in a sentence

        Args:
          dependency_tree: list of tuples (dependent, head) sorted by dependent indicies.
          i: the word in the sentence to compute the depth of
          head_indices: the head indices (according to a dependency parse) of all
              words, or None, if observation != None.

        Returns:
          The integer depth in the tree of word i
        '''
        if dependency_tree:
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
