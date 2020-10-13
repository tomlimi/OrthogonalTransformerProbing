import numpy as np
import tensorflow as tf
from collections import defaultdict

import constants


class ConllWrapper():

    def __init__(self, conll_file, bert_tokenizer, resize_examples):

        self.conllu_name = conll_file
        self.tokenizer = bert_tokenizer

        self.tokens = []
        self.lemmas = []
        self.pos = []
        self.relations = []
        self.roots = []
        self.coreferences = []

        # self.wordpieces = []
        # self.segments = []
        # self.max_segment = []

        self.read_conllu(conll_file)
        self.training_examples(resize_examples)

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

    def resize_index(self, index_to_cut, new_size):
        if self.tokens:
            self.tokens[index_to_cut] = self.tokens[index_to_cut][:new_size]
        if self.lemmas:
            self.lemmas[index_to_cut] = self.lemmas[index_to_cut][:new_size]
        if self.pos:
            self.pos[index_to_cut] = self.pos[index_to_cut][:new_size]
        if self.relations:
            self.relations[index_to_cut] = self.relations[index_to_cut][:new_size]
        if self.roots:
            #TODO: think whether it is the best option
            self.roots[index_to_cut] = min(self.roots[index_to_cut],new_size-1)
        if self.coreferences:
            self.coreferences[index_to_cut] = self.coreferences[index_to_cut][:new_size]

    def read_conllu(self, conll_file_path):
        sentence_relations = []
        sentence_tokens = []
        sentence_lemmas = []
        sentence_pos = []
        sentence_coreference = []


        with open(conll_file_path, 'r') as in_conllu:
            sentid = 0

            curr_coref = set()
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
                    self.coreferences.append(sentence_coreference)
                    sentence_coreference = []
                    coref_counter = defaultdict(int)

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
                        if len(fields) >= constants.CONLL_COREF:
                            coref, curr_coref = self.process_coreference(curr_coref, fields[constants.CONLL_COREF])
                            sentence_coreference.append(coref)

    def get_bert_ids(self, wordpieces):
        """Token ids from Tokenizer vocab"""
        token_ids = self.tokenizer.convert_tokens_to_ids(wordpieces)
        input_ids = token_ids + [0] * (constants.MAX_WORDPIECES - len(wordpieces))
        return input_ids

    def training_examples(self, resize_examples=False):
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

            if resize_examples == True:
                cutting_counter = -1
                while len(sent_tokens) >= constants.MAX_TOKENS or len(sent_wordpieces) >= constants.MAX_WORDPIECES:
                    cutting_counter += 1
                    sent_tokens = sent_tokens[:constants.MAX_TOKENS - 2 - 10*cutting_counter]
                    sent_wordpieces = ["[CLS]"] + self.tokenizer.tokenize((' '.join(sent_tokens)),
                                                                          add_special_tokens=False) + ["[SEP]"]
                else:
                    self.resize_index(idx,constants.MAX_TOKENS - 2 - 10*cutting_counter)

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

        return tf.stack(bert_ids), tf.stack(segments), tf.constant(max_segment, dtype=tf.int64)


