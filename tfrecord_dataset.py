import argparse
import os

import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertModel

import constants

from dependency import DependencyDistance, DependencyDepth
from lexical import LexicalDistance, LexicalDepth


class Dataset:

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _float_features(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _int64_features(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value.reshape(-1)))
    
    class LanguageTaskData:
        def __init__(self, language, task, dependency_data):
            self.language = language
            self.task = task
            self.dependency_data = dependency_data
            # self.target, self.mask = dependency_data.target_and_mask()
            # self.roots = dependency_data.roots
            # self.uu_relations = dependency_data.unlabeled_unordered_relations
            # self.punctuation_mask = dependency_data.punctuation_mask

        @staticmethod
        def serialize_example(target, mask):

            feature = {
                'target': Dataset._float_features(target),
                'mask': Dataset._float_features(mask)
            }

            return tf.train.Example(features=tf.train.Features(feature=feature))

        def write_tfrecord(self):
            filename = f'{self.task}_{self.language}'
            with tf.io.TFRecordWriter(filename) as writer:
                for target, mask in self.dependency_data.target_and_mask():
                    train_example = self.serialize_example(target, mask.numpy())
                    writer.write(train_example.SerializeToString())
        
    class EmbeddedData:
        def __init__(self, language, dependency_data, bert_model):
            self.language = language
            self.dependency_data = dependency_data
            self.bert_model = bert_model
            
            
        def calc_embeddings(self, wordpieces, segments):
            _, _, bert_hidden = self.bert_model(wordpieces, attention_mask=tf.sign(wordpieces), training=False)
            embeddings = bert_hidden[1:]
            # average wordpieces to obtain word representation
            # cut to max nummber of words in batch, note that batch.max_token_len is a tensor, bu all the values are the same
            embeddings = [tf.map_fn(lambda x: tf.math.unsorted_segment_mean(x[0], x[1], x[2]),
                                          (embeddings, segments, constants.MAX_TOKENS), dtype=tf.float32) for embeddings in
                                embeddings]
            return embeddings

        @staticmethod
        def serialize_example(embeddings, token_len):

            feature = {f'layer_{idx}': Dataset._float_feature(layer_embeddings)
                        for idx, layer_embeddings in enumerate(embeddings)}

            feature.update({'num_tokens': Dataset._int64_feature(token_len)})

            return tf.train.Example(features=tf.train.Features(feature=feature))
        
        def write_tfrecord(self):
            filename = f'bert_{self.language}'
            with tf.io.TFRecordWriter(filename) as writer:
                for wordpieces, segments, token_len in zip(self.dependency_data.training_examples()):
                    embeddings = self.calc_embeddings(wordpieces, segments)
                    train_example = self.serialize_example(embeddings, token_len)
                    writer.write(train_example.SerializeToString())
            
    class DatasetWriter:
        def __init__(self, datafiles, languages, tasks, tokenizer, bert_model):

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
                    
                embedding_data = Dataset.EmbeddedData(language, dependency_data, bert_model)
                embedding_data.write_tfrecord()

    def __init__(self, dataset_files, dataset_languages, dataset_tasks, bert_path, do_lower_case=True):
        assert dataset_files.keys() == dataset_languages.keys(), "Specify the same name of datasets."

        tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=do_lower_case)
        bert_model = TFBertModel.from_pretrained(bert_path, output_hidden_states=True)

        for dataset_name in dataset_files.keys():
            self.DatasetWriter(dataset_files[dataset_name],
                               dataset_languages[dataset_name],
                               dataset_tasks[dataset_name],
                               tokenizer,
                               bert_model)
