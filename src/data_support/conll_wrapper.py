import numpy as np
import tensorflow as tf
from collections import defaultdict
import networkx as nx

import constants

RANDOM_SEED = 2019


class ConllWrapper():

    max_wordpieces = None

    def __init__(self, conll_file, bert_tokenizer):

        self.conllu_name = conll_file
        self.tokenizer = bert_tokenizer
        self.random_state = np.random.RandomState(RANDOM_SEED)

        self.tokens = []
        self.lemmas = []
        self.pos = []
        self.relations = []
        self.roots = []
        self.coreferences = []
        self.shuffled = []

        self.read_conllu(conll_file)
        self.training_examples()  # this call is needed here, because it removes too long and mismatched sentences

    @property
    def punctuation_mask(self):
        return [[pos_tag == "PUNCT" for pos_tag in sentence_pos] for sentence_pos in self.pos]

    @property
    def filtered_relations(self):
        return [{(dep, parent) for dep, parent in sent_relation
                 if (not sent_punctuation_mask[dep - 1]) and parent}
                for sent_relation, sent_punctuation_mask in zip(self.relations, self.punctuation_mask)]
    
    @property
    def word_count(self):
        return [len(sent_tokens) for sent_tokens in self.tokens]
    
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
        if self.coreferences:
            self.coreferences = [v for i, v in enumerate(self.coreferences) if i not in indices_to_rm]
        if self.shuffled:
            self.shuffled = [v for i, v in enumerate(self.shuffled) if i not in indices_to_rm]


    def read_conllu(self, conll_file_path):
        sentence_relations = []
        sentence_tokens = []
        sentence_lemmas = []
        sentence_pos = []

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
                        sentence_lemmas.append(fields[constants.CONLLU_LEMMA])
                        sentence_pos.append(fields[constants.CONLLU_POS])

    def get_bert_ids(self, wordpieces):
        """Token ids from Tokenizer vocab"""
        token_ids = self.tokenizer.convert_tokens_to_ids(wordpieces)
        input_ids = token_ids + [0] * (constants.MAX_WORDPIECES - len(wordpieces))
        return input_ids

    def training_examples(self, shuffle=False):
        '''
        Joins wordpices of tokens, so that they correspond to the tokens in conllu file.
        :param shuffle: whether to shuffle tokens in each sentence
        
        :return:
            2-D tensor  [num valid sentences, max num wordpieces] bert wordpiece ids,
            2-D tensor [num valid sentences, max num wordpieces] wordpiece to word segment mappings
            1-D tensor [num valid sentences] number of words in each sentence
        '''
        
        number_examples = len(self.tokens)
        wordpieces = []
        indices_to_rm = []
        
        for idx, sent_tokens in enumerate(self.tokens[:]):
            if shuffle:
                sent_shuf = np.arange(len(sent_tokens))
                self.random_state.shuffle(sent_shuf)
                self.shuffled.append(sent_shuf)
                
                sent_tokens = list(np.array(sent_tokens)[sent_shuf])
                
            sent_wordpieces = [self.tokenizer.cls_token] + self.tokenizer.tokenize((' '.join(sent_tokens))) + [self.tokenizer.sep_token]

            if len(sent_tokens) >= constants.MAX_TOKENS:
                print(f"Sentence {idx} too many tokens, in file {self.conllu_name}, skipping.")
                indices_to_rm.append(idx)
                number_examples -= 1
            elif len(sent_wordpieces) >= constants.MAX_WORDPIECES:
                print(f"Sentence {idx} too many wordpieces, in file {self.conllu_name}, skipping.")
                indices_to_rm.append(idx)
                number_examples -= 1

            wordpieces.append(sent_wordpieces)

        if self.coreferences:
            for idx, sent_corefernces in enumerate(self.coreferences):
                if sum(1 for coref in sent_corefernces if coref) < 2:
                    print(f"Sentence pair {idx} less then two coreferents, in file {self.conllu_name}, skipping.")
                    indices_to_rm.append(idx)

        segments = []
        max_segment = []
        bert_ids = []
        sent_idx = 0
        for sent_wordpieces, sent_tokens in zip(wordpieces, self.tokens[:]):
            if sent_idx in indices_to_rm:
                sent_idx += 1
                continue

            sent_segments = np.zeros((constants.MAX_WORDPIECES,), dtype=np.int64) - 1
            segment_id = 0
            wordpiece_pointer = 1
            if shuffle:
                sent_shuf = self.shuffled[sent_idx]
                sent_tokens = list(np.array(sent_tokens)[sent_shuf])
            for token in sent_tokens:
                worpieces_per_token = len(self.tokenizer.tokenize(token))
                sent_segments[wordpiece_pointer:wordpiece_pointer+worpieces_per_token] = segment_id
                wordpiece_pointer += worpieces_per_token
                segment_id += 1

            if wordpiece_pointer+1 != len(sent_wordpieces):
                print(f'Sentence {sent_idx} mismatch in number of tokens, in file {self.conllu_name}, skipping.')
                indices_to_rm.append(sent_idx)
            else:
                segments.append(tf.constant(sent_segments, dtype=tf.int64))
                bert_ids.append(tf.constant(self.get_bert_ids(sent_wordpieces), dtype=tf.int64))
                max_segment.append(segment_id)
            sent_idx += 1
        self.remove_indices(indices_to_rm)

        return tf.stack(bert_ids), tf.stack(segments), tf.constant(max_segment, dtype=tf.int64)

    def generate_random_tree(self, sentence_length):

        def add_edges(node_idx):
            edges = []
            for dep_node in generated_tree[node_idx]:
                if dep_node not in visited:
                    visited.add(dep_node)
                    edges.append((dep_node + 1, node_idx + 1))
                    edges += add_edges(dep_node)
            return edges

        generated_tree = nx.to_dict_of_lists(nx.generators.trees.random_tree(sentence_length, seed=self.random_state))
        random_root = self.random_state.randint(0, sentence_length)

        visited = {random_root}
        random_tree = [(random_root + 1, 0)] + add_edges(random_root)
        random_tree = sorted(random_tree, key=lambda edge: edge[0])
        return random_tree

