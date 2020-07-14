import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
from scipy import sparse

from metrics import UUAS, RootAcc, Spearman


class Reporter():
	
	def __init__(self, prober, dataset, dataset_name):
		self.prober = prober
		self.dataset = dataset
		self.dataset_name = dataset_name


class DistanceReporter(Reporter):
	
	def __init__(self, prober, dataset, dataset_name):
		super().__init__(prober, dataset, dataset_name)
		self.uuas = dict()
		self.sprarman_d = dict()
	
	def write(self, args):
		for language in self.dataset._languages:
			prefix = '{}.{}.'.format(self.dataset_name, language)
			with open(os.path.join(args.out_dir, prefix + 'uuas'), 'w') as uuas_f:
				uuas_f.write(str(self.uuas[language].result())+'\n')
			
			with open(os.path.join(args.out_dir, prefix + 'spearman'), 'w') as sperarman_f:
				for sent_l, val in self.sprarman_d[language].result().items():
					sperarman_f.write(f'{sent_l}\t{val}\n')
			
			with open(os.path.join(args.out_dir, prefix + 'spearman_mean'), 'w') as sperarman_mean_f:
				result = str(np.nanmean(np.fromiter(self.sprarman_d[language].result().values(), dtype=float)))
				sperarman_mean_f.write(result+'\n')
	
	def predict(self, args):
		for language in self.dataset._languages:
			self.uuas[language] = UUAS()
			self.sprarman_d[language] = Spearman()
			for batch in tqdm(self.dataset.evaluate_batches(language, size=args.batch_size),
			                  desc="Predicting, {}".format(language)):
				predicted = self.prober.predict_on_batch(batch.wordpieces, batch.segments, batch.token_len,
				                                         batch.max_token_len, batch.language)
				predicted = [sent_predicted.numpy()[:sent_len, :sent_len] for sent_predicted, sent_len
				             in zip(tf.unstack(predicted), batch.token_len)]
				
				gold_distances = [sent_gold.numpy()[:sent_len, :sent_len] for sent_gold, sent_len
				                  in zip(tf.unstack(batch.target), batch.token_len)]
				
				self.batch_spearman(language, gold_distances, predicted)
				
				self.batch_uuas(language, gold_distances, predicted, batch.punctuation_mask)
				
	def batch_uuas(self, language, batch_gold, batch_prediction, batch_punctuation_mask):
		# run maximum spanning tree algorithm for each predicted matrix
		for sent_gold, sent_predicted, sent_punctuation_mask in zip(batch_gold, batch_prediction, batch_punctuation_mask):
			sent_predicted[sent_punctuation_mask,:] = np.inf
			sent_predicted[:,sent_punctuation_mask] = np.inf
			min_spanning_tree = sparse.csgraph.minimum_spanning_tree(sent_predicted).tocoo()
			sent_predicted = set(map(frozenset, zip(min_spanning_tree.row + 1, min_spanning_tree.col + 1)))
			
			sent_gold[sent_punctuation_mask,:] = np.inf
			sent_gold[:,sent_punctuation_mask] = np.inf
			min_spanning_tree_gold = sparse.csgraph.minimum_spanning_tree(sent_gold).tocoo()
			sent_gold = set(map(frozenset, zip(min_spanning_tree_gold.row + 1, min_spanning_tree_gold.col + 1)))
			self.uuas[language].update_state(sent_gold, sent_predicted)
	
	def batch_spearman(self, language, batch_gold, batch_prediction):
		
		self.sprarman_d[language](batch_gold, batch_prediction)


class LexicalDistanceReporter(Reporter):

	def __init__(self, prober, dataset, dataset_name):
		super().__init__(prober, dataset, dataset_name)

		self.sprarman_d = dict()

	def write(self, args):
		for language in self.dataset._languages:
			prefix = '{}.{}.'.format(self.dataset_name, language)

			with open(os.path.join(args.out_dir, prefix + 'spearman'), 'w') as sperarman_f:
				for sent_l, val in self.sprarman_d[language].result().items():
					sperarman_f.write(f'{sent_l}\t{val}\n')

			with open(os.path.join(args.out_dir, prefix + 'spearman_mean'), 'w') as sperarman_mean_f:
				result = str(np.nanmean(np.fromiter(self.sprarman_d[language].result().values(), dtype=float)))
				sperarman_mean_f.write(result + '\n')

	def predict(self, args):
		for language in self.dataset._languages:
			self.sprarman_d[language] = Spearman()
			for batch in tqdm(self.dataset.evaluate_batches(language, size=args.batch_size),
							  desc="Predicting, {}".format(language)):
				predicted = self.prober.predict_on_batch(batch.wordpieces, batch.segments, batch.token_len,
														 batch.max_token_len, batch.language)
				predicted = [sent_predicted.numpy()[:sent_len, :sent_len] for sent_predicted, sent_len
							 in zip(tf.unstack(predicted), batch.token_len)]

				gold_distances = [sent_gold.numpy()[:sent_len, :sent_len] for sent_gold, sent_len
								  in zip(tf.unstack(batch.target), batch.token_len)]

				mask = [sent_mask.numpy().astype(bool)[:sent_len,:sent_len] for sent_mask, sent_len
						in zip(tf.unstack(batch.mask), batch.token_len)]

				self.batch_spearman(language, gold_distances, predicted, batch_mask=mask)

	def batch_spearman(self, language, batch_gold, batch_prediction, batch_mask=None):

		self.sprarman_d[language](batch_gold, batch_prediction, batch_mask)


class DepthReporter(Reporter):
	
	def __init__(self, prober, dataset, dataset_name):
		super().__init__(prober, dataset, dataset_name)
		
		self.root_acc = dict()
		self.sprarman_n = dict()
	
	def write(self, args):
		for language in self.dataset._languages:
			prefix = '{}.{}.'.format(self.dataset_name, language)
			with open(os.path.join(args.out_dir, prefix + 'root_acc'), 'w') as root_acc_f:
				root_acc_f.write(str(self.root_acc[language].result()) + '\n')
			
			with open(os.path.join(args.out_dir, prefix + 'spearman'), 'w') as sperarman_f:
				for sent_l, val in self.sprarman_n[language].result().items():
					sperarman_f.write(f'{sent_l}\t{val}\n')
			
			with open(os.path.join(args.out_dir, prefix + 'spearman_mean'), 'w') as sperarman_mean_f:
				result = str(np.nanmean(np.fromiter(self.sprarman_n[language].result().values(), dtype=float)))
				sperarman_mean_f.write(result+'\n')
	
	def predict(self, args):
		for language in self.dataset._languages:
			self.root_acc[language] = RootAcc()
			self.sprarman_n[language] = Spearman()
			for batch in tqdm(self.dataset.evaluate_batches(language, size=args.batch_size),
			                  desc="Predicting, {}".format(language)):
				predicted = self.prober.predict_on_batch(batch.wordpieces, batch.segments, batch.token_len,
				                                         batch.max_token_len, batch.language)
				predicted = [sent_predicted.numpy()[:sent_len] for sent_predicted, sent_len
				             in zip(tf.unstack(predicted), batch.token_len)]
				
				gold_depths = [sent_gold.numpy()[:sent_len] for sent_gold, sent_len
				               in zip(tf.unstack(batch.target), batch.token_len)]
				self.batch_spearman(language, gold_depths, predicted)
				
				self.batch_root_accuracy(language, batch.roots, predicted, batch.punctuation_mask)

	
	def batch_root_accuracy(self, language, batch_gold, batch_prediction, batch_punctution_mask):
		batch_non_punct_prediction = []
		for sent_predicted, sent_punctuation_mask in zip(batch_prediction, batch_punctution_mask):
			sent_predicted[sent_punctuation_mask] = np.inf
			batch_non_punct_prediction.append(sent_predicted)

		self.root_acc[language](batch_gold, [sent_predicted.argmin() + 1 for sent_predicted in batch_non_punct_prediction])
	
	def batch_spearman(self, language, batch_gold, batch_prediction):
		self.sprarman_n[language](batch_gold, batch_prediction)
