import argparse
import os

import tensorflow as tf
import numpy as np
from transformers import BertTokenizer

import constants

from dependency import DependencyDistance, DependencyDepth
from lexical import LexicalDistance, LexicalDepth


class DependencyDataset:
    class LanguageData:

        def __init__(self, language, dependency_data, shuffle_batches=True, seed=42):
            self.language = language
            self.wordpieces, self.segments, self.token_len = dependency_data.training_examples()
            self.target, self.mask = dependency_data.target_and_mask()

            self.roots = dependency_data.roots
            self.uu_relations = dependency_data.unlabeled_unordered_relations
            self.punctuation_mask = dependency_data.punctuation_mask

            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        @property
        def size(self):
            return self.wordpieces.shape[0]

        def get_permutation(self, size=None):
            size = size or self.size
            return self._shuffler.permutation(size) if self._shuffler else np.arange(size)

    class Batch:

        def __init__(self, language_data, batch_indices, task, max_token_len=constants.MAX_TOKENS):
            self.language = language_data.language
            self.wordpieces = tf.gather(language_data.wordpieces, batch_indices)
            self.segments = tf.gather(language_data.segments, batch_indices)
            self.token_len = tf.gather(language_data.token_len, batch_indices)
            self.max_token_len = tf.repeat(max_token_len, tf.size(batch_indices))
            if task.lower() in ("distance", "lex-distance"):
                self.target = tf.gather(language_data.target, batch_indices)[:, :max_token_len, :max_token_len]
                self.mask = tf.gather(language_data.mask, batch_indices)[:, :max_token_len, :max_token_len]
            elif task.lower() in ("depth", "lex-depth"):
                self.target = tf.gather(language_data.target, batch_indices)[:, :max_token_len]
                self.mask = tf.gather(language_data.mask, batch_indices)[:, :max_token_len]
            else:
                raise ValueError("Unknow probing task: {} Choose `depth`, `lex-depth`, `distance` or `lex-distance`".format(task))

            self.roots = [np.array(language_data.roots[exmpl_idx]) for exmpl_idx in batch_indices]
            self.uu_relations = [language_data.uu_relations[exmpl_idx] for exmpl_idx in batch_indices]
            self.punctuation_mask = [np.array(language_data.punctuation_mask[exmpl_idx])
                                     for exmpl_idx in batch_indices]

    class Dataset:
        def __init__(self, datafiles, languages, task, tokenizer, shuffle_batches=True, seed=42):

            self._data = dict()
            self._size = 0
            self._languages = languages
            self._task = task

            for datafile, language in zip(datafiles, languages):
                if task.lower() == "distance":
                    dependency_data = DependencyDistance(datafile, tokenizer)
                elif task.lower() == "depth":
                    dependency_data = DependencyDepth(datafile, tokenizer)
                elif task.lower() == "lex-distance":
                    dependency_data = LexicalDistance(datafile, tokenizer)
                elif task.lower() == 'lex-depth':
                    dependency_data = LexicalDepth(datafile, tokenizer)
                else:
                    raise ValueError(
                        "Unknow probing task: {} Choose `depth`, `lex-depth`, `distance` or `lex-distance`".format(
                            task))

                self._data[language] = DependencyDataset.LanguageData(language, dependency_data,
                                                                      shuffle_batches=shuffle_batches, seed=seed)

                # number of example is equal to number for the language with fewest sentences
                self._size = min([lang_data.size for lang_data in self._data.values()])

        @property
        def data(self):
            return self._data

        @property
        def size(self):
            return self._size

        def train_batches(self, size=None):
            # We stop the training when examples ends for the language with the lowest number of sentences to make
            # the data balanced per language.
            permutations = {language: language_data.get_permutation(self.size)
                            for language, language_data in self._data.items()}
            while any(len(permutation) for permutation in permutations.values()):
                for language in self._languages:
                    if not len(permutations[language]):
                        continue
                    batch_size = min(size or np.inf, len(permutations[language]))
                    batch_perm = permutations[language][:batch_size]
                    permutations[language] = permutations[language][batch_size:]

                    batch_indices = tf.constant(batch_perm)
                    max_token_len = tf.reduce_max(tf.gather(self._data[language].token_len, batch_indices))
                    batch = DependencyDataset.Batch(self._data[language], batch_indices, self._task, max_token_len)
                    yield batch

        def evaluate_batches(self, language, size=None):
            permutation = self._data[language].get_permutation()
            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                batch_indices = tf.constant(batch_perm)
                max_token_len = tf.reduce_max(tf.gather(self._data[language].token_len, batch_indices))
                batch = DependencyDataset.Batch(self._data[language], batch_indices, self._task, max_token_len)

                yield batch

    def __init__(self, dataset_files, dataset_languages, task, bert_path, do_lower_case=True):
        assert dataset_files.keys() == dataset_languages.keys(), "Specify the same name of datasets."

        tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=do_lower_case)

        for dataset_name in dataset_files.keys():
            setattr(self, dataset_name, self.Dataset(dataset_files[dataset_name],
                                                     dataset_languages[dataset_name],
                                                     task,
                                                     tokenizer))
