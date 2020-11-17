import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
from scipy import sparse
from collections import defaultdict

from network import Network

from reporting.metrics import UAS, Spearman


class Reporter():

	def __init__(self, network, dataset, dataset_name):
		self.network = network
		self.dataset = dataset
		self.dataset_name = dataset_name


class CorrelationReporter(Reporter):

	def __init__(self, args, network, dataset, dataset_name):
		super().__init__(network, dataset, dataset_name)

		self._languages = args.languages
		self._tasks = args.tasks
		self.spearman_d = defaultdict(dict)

	def write(self, args):
		for language in self._languages:
			for task in self._tasks:
				prefix = '{}.{}.{}.'.format(self.dataset_name, language, task)

				with open(os.path.join(args.out_dir, prefix + 'spearman'), 'w') as sperarman_f:
					for sent_l, val in self.spearman_d[language][task].result().items():
						sperarman_f.write(f'{sent_l}\t{val}\n')

				with open(os.path.join(args.out_dir, prefix + 'spearman_mean'), 'w') as sperarman_mean_f:
					result = str(np.nanmean(np.fromiter(self.spearman_d[language][task].result().values(), dtype=float)))
					sperarman_mean_f.write(result + '\n')

	def predict(self, args):

		test = {lang: {task: Network.data_pipeline(self.dataset, [lang], [task], args, mode='test')
		               for task in self._tasks}
		        for lang in self._languages}

		for language in self._languages:
			for task in self._tasks:

				self.spearman_d[language][task] = Spearman()
				progressbar = tqdm(enumerate(test[language][task]), desc="Predicting, {}, {}".format(language, task))
				for batch_idx, (_, _, batch) in progressbar:
					_, batch_target, batch_mask, batch_num_tokens, batch_embeddings = batch
					sent_lens = list(batch_num_tokens.numpy())
					if 'distance' in task:
						pred_values = self.network.distance_probe.predict_on_batch(batch_num_tokens, batch_embeddings,
						                                                           language, task)
						pred_values = [sent_predicted.numpy()[:sent_len, :sent_len] for sent_predicted, sent_len
						               in zip(tf.unstack(pred_values), sent_lens)]
						gold_values = [sent_gold.numpy()[:sent_len, :sent_len] for sent_gold, sent_len
						               in zip(tf.unstack(batch_target), sent_lens)]
						mask = [sent_mask.numpy().astype(bool)[:sent_len, :sent_len] for sent_mask, sent_len
						        in zip(tf.unstack(batch_mask), sent_lens)]
					elif 'depth' in task:
						pred_values = self.network.depth_probe.predict_on_batch(batch_num_tokens, batch_embeddings,
						                                                        language, task)
						pred_values = [sent_predicted.numpy()[:sent_len] for sent_predicted, sent_len
						               in zip(tf.unstack(pred_values), sent_lens)]
						gold_values = [sent_gold.numpy()[:sent_len] for sent_gold, sent_len
						               in zip(tf.unstack(batch_target), sent_lens)]
						mask = [sent_mask.numpy().astype(bool)[:sent_len] for sent_mask, sent_len
						        in zip(tf.unstack(batch_mask), sent_lens)]

					self.spearman_d[language][task](gold_values, pred_values, mask)


class GatedCorrelationReporter(Reporter):

	def __init__(self, args, network, dataset, dataset_name):
		super().__init__(network, dataset, dataset_name)

		self._probe_threshold = args.probe_threshold
		self._drop_parts = args.drop_parts

		self._languages = args.languages
		self._tasks = args.tasks
		self.spearman_d = defaultdict(dict)

	def write(self, args):
		for language in self._languages:
			for task in self._tasks:
				prefix = '{}.{}.{}.gated.'.format(self.dataset_name, language, task)
				if self._drop_parts:
					prefix += 'dp{}.'.format(self._drop_parts)

				with open(os.path.join(args.out_dir, prefix + 'spearman'), 'w') as sperarman_f:
					for sent_l, val in self.spearman_d[language][task].result().items():
						sperarman_f.write(f'{sent_l}\t{val}\n')

				with open(os.path.join(args.out_dir, prefix + 'spearman_mean'), 'w') as sperarman_mean_f:
					result = str(np.nanmean(np.fromiter(self.spearman_d[language][task].result().values(), dtype=float)))
					sperarman_mean_f.write(result + '\n')

	def get_embedding_gate(self, task, part_to_drop):

		if 'distance' in task:
			diagonal_probe = self.network.distance_probe.DistanceProbe[task].numpy()
		elif 'depth' in task:
			diagonal_probe = self.network.depth_probe.DepthProbe[task].numpy()
		embedding_gate = (np.abs(diagonal_probe) > self._probe_threshold).astype(np.float)

		if self._drop_parts:
			dim_num = np.sum(embedding_gate)
			part_start = int(dim_num * part_to_drop / self._drop_parts)
			part_end = int(dim_num * (part_to_drop+1) / self._drop_parts)
			dims_to_drop = np.where(embedding_gate)[-1][part_start:part_end]
			embedding_gate[...,dims_to_drop] = 0.

		return tf.constant(embedding_gate, dtype=tf.float32)



	def predict(self, args):

		test = {lang: {task: Network.data_pipeline(self.dataset, [lang], [task], args, mode='test')
		               for task in self._tasks}
		        for lang in self._languages}

		validation_steps = self._drop_parts or 1

		for language in self._languages:
			for task in self._tasks:
				self.spearman_d[language][task] = Spearman()

				for part_to_drop in range(validation_steps):
					progressbar = tqdm(enumerate(test[language][task]), desc="Predicting, {}, {}".format(language, task))
					for batch_idx, (_, _, batch) in progressbar:
						_, batch_target, batch_mask, batch_num_tokens, batch_embeddings = batch

						embedding_gate = self.get_embedding_gate(task, part_to_drop)
						sent_lens = list(batch_num_tokens.numpy())
						if 'distance' in task:
							pred_values = self.network.distance_probe.predict_on_batch(batch_num_tokens, batch_embeddings,
							                                                           language, task, embedding_gate)
							pred_values = [sent_predicted.numpy()[:sent_len, :sent_len] for sent_predicted, sent_len
							               in zip(tf.unstack(pred_values), sent_lens)]
							gold_values = [sent_gold.numpy()[:sent_len, :sent_len] for sent_gold, sent_len
							               in zip(tf.unstack(batch_target), sent_lens)]
							mask = [sent_mask.numpy().astype(bool)[:sent_len, :sent_len] for sent_mask, sent_len
							        in zip(tf.unstack(batch_mask), sent_lens)]
						elif 'depth' in task:
							pred_values = self.network.depth_probe.predict_on_batch(batch_num_tokens, batch_embeddings,
							                                                        language, task, embedding_gate)
							pred_values = [sent_predicted.numpy()[:sent_len] for sent_predicted, sent_len
							               in zip(tf.unstack(pred_values), sent_lens)]
							gold_values = [sent_gold.numpy()[:sent_len] for sent_gold, sent_len
							               in zip(tf.unstack(batch_target), sent_lens)]
							mask = [sent_mask.numpy().astype(bool)[:sent_len] for sent_mask, sent_len
							        in zip(tf.unstack(batch_mask), sent_lens)]

						self.spearman_d[language][task](gold_values, pred_values, mask)


class UASReporter(Reporter):
	def __init__(self, args, network, dataset, dataset_name, conll_dict):
		super().__init__(network, dataset, dataset_name)
		self.punctuation_masks = {lang: conll_data.punctuation_mask for lang, conll_data in conll_dict.items()}
		self.uu_rels = {lang: conll_data.unlabeled_unordered_relations for lang, conll_data in conll_dict.items()}
		self._languages = args.languages
		self._tasks = args.tasks
		self.uas = dict()

	def write(self, args):
		for language in self._languages:
			prefix = '{}.{}.'.format(self.dataset_name, language)
			with open(os.path.join(args.out_dir, prefix + 'uas'), 'w') as uuas_f:
				uuas_f.write(str(self.uas[language].result())+'\n')

	def predict(self, args):
		test = {lang: Network.data_pipeline(self.dataset, [lang], ['dep_distance'], args, mode='test')
		        for lang in self._languages}

		for language in self._languages:
			self.uas[language] = UAS()
			progressbar = tqdm(enumerate(test[language]), desc="Predicting UAS, {}".format(language))
			for batch_idx, (_, _, batch) in progressbar:
				conll_indices, batch_target, batch_mask, batch_num_tokens, batch_embeddings = batch

				sent_lens = list(batch_num_tokens.numpy())
				pred_values = self.network.distance_probe.predict_on_batch(batch_num_tokens, batch_embeddings,
				                                                           language, 'dep_distance')
				pred_values = [sent_predicted.numpy()[:sent_len, :sent_len] for sent_predicted, sent_len
				               in zip(tf.unstack(pred_values), sent_lens)]
				gold_distances = [sent_gold.numpy()[:sent_len, :sent_len] for sent_gold, sent_len
				                  in zip(tf.unstack(batch_target), sent_lens)]

				conll_indices = conll_indices.numpy()

				for conll_idx, sent_predicted, sent_gold in zip(conll_indices, pred_values, gold_distances):

					sent_punctuation_mask = self.punctuation_masks[language][conll_idx]
					sent_predicted[sent_punctuation_mask, :] = np.inf
					sent_predicted[:, sent_punctuation_mask] = np.inf
					min_spanning_tree = sparse.csgraph.minimum_spanning_tree(sent_predicted).tocoo()
					predicted = set(map(frozenset, zip(min_spanning_tree.row + 1, min_spanning_tree.col + 1)))

					sent_gold[sent_punctuation_mask, :] = np.inf
					sent_gold[:, sent_punctuation_mask] = np.inf
					min_spanning_tree_gold = sparse.csgraph.minimum_spanning_tree(sent_gold).tocoo()
					gold = set(map(frozenset, zip(min_spanning_tree_gold.row + 1, min_spanning_tree_gold.col + 1)))
					#gold = set(self.uu_rels[language][conll_idx])


					self.uas[language].update_state(gold, predicted)


# class DependencyDistanceReporter(Reporter):
#
# 	def __init__(self, prober, dataset, dataset_name):
# 		super().__init__(prober, dataset, dataset_name)
# 		self.uuas = dict()
# 		self.sprarman_d = dict()
#
# 	def write(self, args):
# 		for language in self.dataset._languages:
# 			prefix = '{}.{}.'.format(self.dataset_name, language)
# 			with open(os.path.join(args.out_dir, prefix + 'uuas'), 'w') as uuas_f:
# 				uuas_f.write(str(self.uuas[language].result())+'\n')
#
# 			with open(os.path.join(args.out_dir, prefix + 'spearman'), 'w') as sperarman_f:
# 				for sent_l, val in self.sprarman_d[language].result().items():
# 					sperarman_f.write(f'{sent_l}\t{val}\n')
#
# 			with open(os.path.join(args.out_dir, prefix + 'spearman_mean'), 'w') as sperarman_mean_f:
# 				result = str(np.nanmean(np.fromiter(self.sprarman_d[language].result().values(), dtype=float)))
# 				sperarman_mean_f.write(result+'\n')
#
# 	def predict(self, args):
# 		#TODO: change prediction
# 		for language in self.dataset._languages:
# 			self.uuas[language] = UUAS()
# 			self.sprarman_d[language] = Spearman()
# 			for batch in tqdm(self.dataset.evaluate_batches(language, size=args.batch_size),
# 			                  desc="Predicting, {}".format(language)):
#
#
# 				predicted = self.prober.predict_on_batch(batch.wordpieces, batch.segments, batch.token_len,
# 				                                         batch.max_token_len, batch.language)
#
# 				predicted = [sent_predicted.numpy()[:sent_len, :sent_len] for sent_predicted, sent_len
# 				             in zip(tf.unstack(predicted), batch.token_len)]
#
# 				gold_distances = [sent_gold.numpy()[:sent_len, :sent_len] for sent_gold, sent_len
# 				                  in zip(tf.unstack(batch.target), batch.token_len)]
#
# 				self.sprarman_d[language](gold_distances, predicted)
#
# 				self.batch_uuas(language, gold_distances, predicted, batch.punctuation_mask)
#
# 	def batch_uuas(self, language, batch_gold, batch_prediction, batch_punctuation_mask):
# 		# run maximum spanning tree algorithm for each predicted matrix
# 		for sent_gold, sent_predicted, sent_punctuation_mask in zip(batch_gold, batch_prediction, batch_punctuation_mask):
# 			sent_predicted[sent_punctuation_mask,:] = np.inf
# 			sent_predicted[:,sent_punctuation_mask] = np.inf
# 			min_spanning_tree = sparse.csgraph.minimum_spanning_tree(sent_predicted).tocoo()
# 			sent_predicted = set(map(frozenset, zip(min_spanning_tree.row + 1, min_spanning_tree.col + 1)))
#
# 			sent_gold[sent_punctuation_mask,:] = np.inf
# 			sent_gold[:,sent_punctuation_mask] = np.inf
# 			min_spanning_tree_gold = sparse.csgraph.minimum_spanning_tree(sent_gold).tocoo()
# 			sent_gold = set(map(frozenset, zip(min_spanning_tree_gold.row + 1, min_spanning_tree_gold.col + 1)))
# 			self.uuas[language].update_state(sent_gold, sent_predicted)
#
#
# class LexicalDistanceReporter(Reporter):
#
# 	def __init__(self, prober, dataset, dataset_name):
# 		super().__init__(prober, dataset, dataset_name)
#
# 		self.sprarman_d = dict()
#
# 	def write(self, args):
# 		for language in self.dataset._languages:
# 			prefix = '{}.{}.'.format(self.dataset_name, language)
#
# 			with open(os.path.join(args.out_dir, prefix + 'spearman'), 'w') as sperarman_f:
# 				for sent_l, val in self.sprarman_d[language].result().items():
# 					sperarman_f.write(f'{sent_l}\t{val}\n')
#
# 			with open(os.path.join(args.out_dir, prefix + 'spearman_mean'), 'w') as sperarman_mean_f:
# 				result = str(np.nanmean(np.fromiter(self.sprarman_d[language].result().values(), dtype=float)))
# 				sperarman_mean_f.write(result + '\n')
#
# 	def predict(self, args):
# 		for language in self.dataset._languages:
# 			self.sprarman_d[language] = Spearman()
# 			for batch in tqdm(self.dataset.evaluate_batches(language, size=args.batch_size),
# 			                  desc="Predicting, {}".format(language)):
# 				predicted = self.prober.predict_on_batch(batch.wordpieces, batch.segments, batch.token_len,
# 				                                         batch.max_token_len, batch.language)
# 				predicted = [sent_predicted.numpy()[:sent_len, :sent_len] for sent_predicted, sent_len
# 				             in zip(tf.unstack(predicted), batch.token_len)]
#
# 				gold_distances = [sent_gold.numpy()[:sent_len, :sent_len] for sent_gold, sent_len
# 				                  in zip(tf.unstack(batch.target), batch.token_len)]
#
# 				mask = [sent_mask.numpy().astype(bool)[:sent_len,:sent_len] for sent_mask, sent_len
# 				        in zip(tf.unstack(batch.mask), batch.token_len)]
#
# 				self.sprarman_d[language](gold_distances, predicted, mask)
#
#
# class DependencyDepthReporter(Reporter):
#
# 	def __init__(self, prober, dataset, dataset_name):
# 		super().__init__(prober, dataset, dataset_name)
#
# 		self.root_acc = dict()
# 		self.sprarman_n = dict()
#
# 	def write(self, args):
# 		for language in self.dataset._languages:
# 			prefix = '{}.{}.'.format(self.dataset_name, language)
# 			with open(os.path.join(args.out_dir, prefix + 'root_acc'), 'w') as root_acc_f:
# 				root_acc_f.write(str(self.root_acc[language].result()) + '\n')
#
# 			with open(os.path.join(args.out_dir, prefix + 'spearman'), 'w') as sperarman_f:
# 				for sent_l, val in self.sprarman_n[language].result().items():
# 					sperarman_f.write(f'{sent_l}\t{val}\n')
#
# 			with open(os.path.join(args.out_dir, prefix + 'spearman_mean'), 'w') as sperarman_mean_f:
# 				result = str(np.nanmean(np.fromiter(self.sprarman_n[language].result().values(), dtype=float)))
# 				sperarman_mean_f.write(result+'\n')
#
# 	def predict(self, args):
# 		for language in self.dataset._languages:
# 			self.root_acc[language] = RootAcc()
# 			self.sprarman_n[language] = Spearman()
# 			for batch in tqdm(self.dataset.evaluate_batches(language, size=args.batch_size),
# 			                  desc="Predicting, {}".format(language)):
# 				predicted = self.prober.predict_on_batch(batch.wordpieces, batch.segments, batch.token_len,
# 				                                         batch.max_token_len, batch.language)
# 				predicted = [sent_predicted.numpy()[:sent_len] for sent_predicted, sent_len
# 				             in zip(tf.unstack(predicted), batch.token_len)]
#
# 				gold_depths = [sent_gold.numpy()[:sent_len] for sent_gold, sent_len
# 				               in zip(tf.unstack(batch.target), batch.token_len)]
# 				self.sprarman_n[language](gold_depths, predicted)
# 				self.batch_root_accuracy(language, batch.roots, predicted, batch.punctuation_mask)
#
#
# 	def batch_root_accuracy(self, language, batch_gold, batch_prediction, batch_punctution_mask):
# 		batch_non_punct_prediction = []
# 		for sent_predicted, sent_punctuation_mask in zip(batch_prediction, batch_punctution_mask):
# 			sent_predicted[sent_punctuation_mask] = np.inf
# 			batch_non_punct_prediction.append(sent_predicted)
#
# 		self.root_acc[language](batch_gold, [sent_predicted.argmin() + 1 for sent_predicted in batch_non_punct_prediction])
#
#
# class LexicalDepthReporter(Reporter):
#
# 	def __init__(self, prober, dataset, dataset_name):
# 		super().__init__(prober, dataset, dataset_name)
#
# 		self.sprarman_n = dict()
#
# 	def write(self, args):
# 		for language in self.dataset._languages:
# 			prefix = '{}.{}.'.format(self.dataset_name, language)
#
#
# 			with open(os.path.join(args.out_dir, prefix + 'spearman'), 'w') as sperarman_f:
# 				for sent_l, val in self.sprarman_n[language].result().items():
# 					sperarman_f.write(f'{sent_l}\t{val}\n')
#
# 			with open(os.path.join(args.out_dir, prefix + 'spearman_mean'), 'w') as sperarman_mean_f:
# 				result = str(np.nanmean(np.fromiter(self.sprarman_n[language].result().values(), dtype=float)))
# 				sperarman_mean_f.write(result + '\n')
#
# 	def predict(self, args):
# 		for language in self.dataset._languages:
# 			self.sprarman_n[language] = Spearman()
# 			for batch in tqdm(self.dataset.evaluate_batches(language, size=args.batch_size),
# 			                  desc="Predicting, {}".format(language)):
# 				predicted = self.prober.predict_on_batch(batch.wordpieces, batch.segments, batch.token_len,
# 				                                         batch.max_token_len, batch.language)
# 				predicted = [sent_predicted.numpy()[:sent_len] for sent_predicted, sent_len
# 				             in zip(tf.unstack(predicted), batch.token_len)]
#
# 				gold_depths = [sent_gold.numpy()[:sent_len] for sent_gold, sent_len
# 				               in zip(tf.unstack(batch.target), batch.token_len)]
#
# 				mask = [sent_mask.numpy().astype(bool)[:sent_len] for sent_mask, sent_len
# 				        in zip(tf.unstack(batch.mask), batch.token_len)]
#
# 				self.sprarman_n[language](gold_depths, predicted, mask)
