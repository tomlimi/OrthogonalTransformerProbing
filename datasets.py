import argparse
import os

import tensorflow as tf
import numpy as np
import bert

import constants

from dependency import DependencyDistance, DependencyDepth


class DependencyDataset:
	
	class LanguageData:
		
		def __init__(self, language, dependency_data):
			self.language = language
			self.wordpieces, self.segments, self.token_len = dependency_data.training_examples()
			self.target = dependency_data.target_tensor()
			
			self.roots = dependency_data.roots
			self.uu_relations = dependency_data.unlabeled_unordered_relations
			
		@property
		def size(self):
			return self.wordpieces.shape[0]
			
	class Batch:
		
		def __init__(self, language_data, batch_indices, task, max_token_len=constants.MAX_TOKENS):
			self.language = language_data.language
			self.wordpieces = tf.gather(language_data.wordpieces, batch_indices)
			self.segments = tf.gather(language_data.segments, batch_indices)
			self.token_len = tf.gather(language_data.token_len, batch_indices)
			self.max_token_len = tf.repeat(max_token_len, tf.size(batch_indices))
			if task.lower() == "distance":
				self.target = tf.gather(language_data.target, batch_indices)[:,:max_token_len,:max_token_len]
			elif task.lower() == "depth":
				self.target = tf.gather(language_data.target, batch_indices)[:,:max_token_len]
			else:
				raise ValueError("Unknow probing task: {} Choose `depth` or `distance`".format(task))
			
			self.roots = [np.array(language_data.roots[exmpl_idx]) for exmpl_idx in batch_indices]
			self.uu_relations = [language_data.uu_relations[exmpl_idx] for exmpl_idx in batch_indices]
			
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
				else:
					raise ValueError("Unknow probing task: {} Choose `depth` or `distance`".format(task))
				
				self._data[language] = DependencyDataset.LanguageData(language, dependency_data)
				
			# number of example is equal to number for the language with fewest sentences
			self._size = min([lang_data.size for lang_data in self._data.values()])
			self._shuffler = np.random.RandomState(seed) if shuffle_batches else None
		
		@property
		def data(self):
			return self._data
	
		@property
		def size(self):
			return self._size
	
		def train_batches(self, size=None):
			permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
			while len(permutation):
				batch_size = min(size or np.inf, len(permutation))
				batch_perm = permutation[:batch_size]
				permutation = permutation[batch_size:]
				for language in self._languages:
					batch_indices = tf.constant(batch_perm)
					max_token_len = tf.reduce_max(tf.gather(self._data[language].token_len, batch_indices))
					batch = DependencyDataset.Batch(self._data[language], batch_indices, self._task, max_token_len)
					yield batch
				
		def evaluate_batches(self, language, size=None):
			permutation = np.arange(self._size)
			while len(permutation):
				batch_size = min(size or np.inf, len(permutation))
				batch_perm = permutation[:batch_size]
				permutation = permutation[batch_size:]
	
				batch_indices = tf.constant(batch_perm)
				max_token_len = tf.reduce_max(tf.gather(self._data[language].token_len, batch_indices))
				batch = DependencyDataset.Batch(self._data[language], batch_indices,self._task, max_token_len)
				
				yield batch

	def __init__(self, dataset_files, dataset_languages, task, bert_dir, do_lower_case=True):
		assert dataset_files.keys() == dataset_languages.keys(), "Specify the same name of datasets."
		
		vocab_file = os.path.join(bert_dir, "vocab.txt")
		tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
		
		for dataset_name in dataset_files.keys():
			setattr(self, dataset_name, self.Dataset(dataset_files[dataset_name],
			                                         dataset_languages[dataset_name],
			                                         task,
			                                         tokenizer))
