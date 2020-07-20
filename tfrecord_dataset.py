import argparse
import os

import tensorflow as tf
import numpy as np
from transformers import BertTokenizer

import constants

from dependency import DependencyDistance, DependencyDepth
from lexical import LexicalDistance, LexicalDepth


class Dataset:
    def create_float_feature(values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))
    
    def create_int_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    
    class LanguageTaskData:
        def __init__(self, language, task, dependency_data):
            self.language = language
            self.task = task
            
            self.target, self.mask = dependency_data.target_and_mask()
            self.roots = dependency_data.roots
            self.uu_relations = dependency_data.unlabeled_unordered_relations
            self.punctuation_mask = dependency_data.punctuation_mask


        def write_tfrecord(self):
            raise NotImplementedError
            
        
    class FeaturesData:
        def __init__(self, language, dependency_data):
            self.wordpieces, self.segments, self.token_len = dependency_data.training_examples()
            
            
        def calc_embeddings(self, bert_model, wordpieces, segments, max_token_len):
            _, _, bert_hidden = bert_model(wordpieces, attention_mask=tf.sign(wordpieces), training=False)
            layer_embeddings = bert_hidden[1:]
            # average wordpieces to obtain word representation
            # cut to max nummber of words in batch, note that batch.max_token_len is a tensor, bu all the values are the same
            layer_embeddings = [tf.map_fn(lambda x: tf.math.unsorted_segment_mean(x[0], x[1], x[2]),
                                          (embeddings, segments, max_token_len), dtype=tf.float32) for embeddings in
                                layer_embeddings]
            return layer_embeddings
        
        def write_tfrecord(self):
            raise NotImplementedError
            
    class DatasetWriter:
        def __init__(self, datafiles, languages, tasks, tokenizer):

            self._data = dict()
            self._size = 0
            self._languages = languages
            self._tasks = tasks

            for datafile, language in zip(datafiles, languages):
                for task in tasks:
                    if task.lower() == "distance":
                        dependency_data = DependencyDistance(datafile, tokenizer)
                    elif task.lower() == "depth":
                        dependency_data = DependencyDepth(datafile, tokenizer)
                    elif task.lower() == "lex-distance":
                        dependency_data = LexicalDistance(datafile, tokenizer, lang=language)
                    elif task.lower() == 'lex-depth':
                        dependency_data = LexicalDepth(datafile, tokenizer, lang=language)
                    else:
                        raise ValueError(
                            "Unknow probing task: {} Choose `depth`, `lex-depth`, `distance` or `lex-distance`".format(
                                task))

                    language_data = Dataset.LanguageTaskData(language, task, dependency_data)
                    language_data.write_tfrecord()
                    
                model_data = Dataset.FeaturesData(language, dependency_data)
                model_data.write_tfrecord()
                model_data.write_tfrecord()

    def __init__(self, dataset_files, dataset_languages, task, bert_path, do_lower_case=True):
        assert dataset_files.keys() == dataset_languages.keys(), "Specify the same name of datasets."

        tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=do_lower_case)

        for dataset_name in dataset_files.keys():
            setattr(self, dataset_name, self.DatasetWriter(dataset_files[dataset_name],
                                                     dataset_languages[dataset_name],
                                                     task,
                                                     tokenizer))
