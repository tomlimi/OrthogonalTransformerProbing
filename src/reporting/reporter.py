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

		test = {lang: {task: Network.data_pipeline(self.dataset, [lang], [task], args, mode=self.dataset_name)
		               for task in self._tasks}
		        for lang in self._languages}

		for language in self._languages:
			for task in self._tasks:

				self.spearman_d[language][task] = Spearman()
				progressbar = tqdm(enumerate(test[language][task]), desc="Predicting, {}, {}".format(language, task))
				for batch_idx, (_, _, batch) in progressbar:
					_, batch_target, batch_mask, batch_num_tokens, batch_embeddings = batch
					#batch_num_tokens = list(batch_num_tokens.numpy())
					if 'distance' in task:
						pred_values = self.network.distance_probe.predict_on_batch(batch_num_tokens, batch_embeddings,
						                                                           language, task)
						pred_values = [sent_predicted.numpy()[:sent_len, :sent_len] for sent_predicted, sent_len
						               in zip(tf.unstack(pred_values), batch_num_tokens)]
						gold_values = [sent_gold.numpy()[:sent_len, :sent_len] for sent_gold, sent_len
						               in zip(tf.unstack(batch_target), batch_num_tokens)]
						mask = [sent_mask.numpy().astype(bool)[:sent_len, :sent_len] for sent_mask, sent_len
						        in zip(tf.unstack(batch_mask), batch_num_tokens)]
					elif 'depth' in task:
						pred_values = self.network.depth_probe.predict_on_batch(batch_num_tokens, batch_embeddings,
						                                                        language, task)
						pred_values = [sent_predicted.numpy()[:sent_len] for sent_predicted, sent_len
						               in zip(tf.unstack(pred_values), batch_num_tokens)]
						gold_values = [sent_gold.numpy()[:sent_len] for sent_gold, sent_len
						               in zip(tf.unstack(batch_target), batch_num_tokens)]
						mask = [sent_mask.numpy().astype(bool)[:sent_len] for sent_mask, sent_len
						        in zip(tf.unstack(batch_mask), batch_num_tokens)]

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

		test = {lang: {task: Network.data_pipeline(self.dataset, [lang], [task], args, mode=self.dataset_name)
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
						# batch_num_tokens = list(batch_num_tokens.numpy())
						if 'distance' in task:
							pred_values = self.network.distance_probe.predict_on_batch(batch_num_tokens, batch_embeddings,
							                                                           language, task, embedding_gate)
							pred_values = [sent_predicted.numpy()[:sent_len, :sent_len] for sent_predicted, sent_len
							               in zip(tf.unstack(pred_values), batch_num_tokens)]
							gold_values = [sent_gold.numpy()[:sent_len, :sent_len] for sent_gold, sent_len
							               in zip(tf.unstack(batch_target), batch_num_tokens)]
							mask = [sent_mask.numpy().astype(bool)[:sent_len, :sent_len] for sent_mask, sent_len
							        in zip(tf.unstack(batch_mask), batch_num_tokens)]
						elif 'depth' in task:
							pred_values = self.network.depth_probe.predict_on_batch(batch_num_tokens, batch_embeddings,
							                                                        language, task, embedding_gate)
							pred_values = [sent_predicted.numpy()[:sent_len] for sent_predicted, sent_len
							               in zip(tf.unstack(pred_values), batch_num_tokens)]
							gold_values = [sent_gold.numpy()[:sent_len] for sent_gold, sent_len
							               in zip(tf.unstack(batch_target), batch_num_tokens)]
							mask = [sent_mask.numpy().astype(bool)[:sent_len] for sent_mask, sent_len
							        in zip(tf.unstack(batch_mask), batch_num_tokens)]

						self.spearman_d[language][task](gold_values, pred_values, mask)


class UASReporter(Reporter):
	def __init__(self, args, network, dataset, dataset_name, conll_dict, depths=None):
		super().__init__(network, dataset, dataset_name)
		self.punctuation_masks = {lang: conll_data.punctuation_mask for lang, conll_data in conll_dict.items()}
		self.uu_rels = {lang: conll_data.filtered_relations for lang, conll_data in conll_dict.items()}
		self._languages = args.languages
		self.uas = dict()

		self.depths = depths

	def write(self, args):
		for language in self._languages:
			prefix = '{}.{}.'.format(self.dataset_name, language)
			if self.depths:
				with open(os.path.join(args.out_dir, prefix + 'uas'), 'w') as uas_f:
					uas_f.write(str(self.uas[language].result())+'\n')
			else:
				with open(os.path.join(args.out_dir, prefix + 'uuas'), 'w') as uuas_f:
					uuas_f.write(str(self.uas[language].result())+'\n')

	def predict(self, args):
		test = {lang: Network.data_pipeline(self.dataset, [lang], ['dep_distance'], args, mode=self.dataset_name)
		        for lang in self._languages}

		for language in self._languages:
			self.uas[language] = UAS()
			progressbar = tqdm(enumerate(test[language]), desc="Predicting UAS, {}".format(language))
			for batch_idx, (_, _, batch) in progressbar:
				conll_indices, batch_target, batch_mask, batch_num_tokens, batch_embeddings = batch

				#batch_num_tokens = list(batch_num_tokens.numpy())
				pred_values = self.network.distance_probe.predict_on_batch(batch_num_tokens, batch_embeddings,
				                                                           language, 'dep_distance')
				pred_values = [sent_predicted.numpy()[:sent_len, :sent_len] for sent_predicted, sent_len
				               in zip(tf.unstack(pred_values), batch_num_tokens)]
				gold_distances = [sent_gold.numpy()[:sent_len, :sent_len] for sent_gold, sent_len
				                  in zip(tf.unstack(batch_target), batch_num_tokens)]

				conll_indices = conll_indices.numpy()

				for conll_idx, sent_predicted, sent_gold, sent_len in zip(conll_indices, pred_values,
				                                                          gold_distances, batch_num_tokens):
					sent_punctuation_mask = self.punctuation_masks[language][conll_idx]
					if self.depths:
						predicted_depths = self.depths[language][conll_idx]["predicted"]
						gold_depths = self.depths[language][conll_idx]["gold"]
						predicted_root = np.argmin(predicted_depths) + 1
						gold_root = np.argmin(gold_depths) + 1

					for i in range(sent_len):
						for j in range(sent_len):
							if i in sent_punctuation_mask or j in sent_punctuation_mask:
								sent_predicted[i, j] = np.inf
								sent_gold[i, j] = np.inf
							elif self.depths is None:
								if i > j:
									sent_predicted[i, j] = np.inf
									sent_gold[i, j] = np.inf
							else:
								if predicted_depths[i] > predicted_depths[j]:
									sent_predicted[i, j] = np.inf
								if gold_depths[i] > gold_depths[j]:
									sent_gold[i, j] = np.inf

					min_spanning_tree = sparse.csgraph.minimum_spanning_tree(sent_predicted).tocoo()
					min_spanning_tree_gold = sparse.csgraph.minimum_spanning_tree(sent_gold).tocoo()

					predicted = set(map(tuple, zip(min_spanning_tree.col + 1, min_spanning_tree.row + 1)))
					gold = set(map(tuple, zip(min_spanning_tree_gold.col + 1, min_spanning_tree_gold.row + 1)))

					if self.depths:
						predicted.add((predicted_root, 0))
						gold.add((gold_root, 0))

					self.uas[language].update_state(gold, predicted)


def predict_dep_depths(args, network, dataset, dataset_name):
	languages = args.languages

	test = {lang: Network.data_pipeline(dataset, [lang], ['dep_depth'], args, mode=dataset_name)
	        for lang in languages}

	results = {lang: dict() for lang in languages}

	for language in languages:
		progressbar = tqdm(enumerate(test[language]), desc="Predicting dependency depths, {}".format(language))
		for batch_idx, (_, _, batch) in progressbar:
			conll_indices, batch_target, batch_mask, batch_num_tokens, batch_embeddings = batch

			pred_values = network.depth_probe.predict_on_batch(batch_num_tokens, batch_embeddings, language, 'dep_depth')
			pred_values = [sent_predicted.numpy()[:sent_len] for sent_predicted, sent_len
			               in zip(tf.unstack(pred_values), batch_num_tokens)]

			gold_depths = [sent_gold.numpy()[:sent_len] for sent_gold, sent_len
			                  in zip(tf.unstack(batch_target), batch_num_tokens)]

			conll_indices = conll_indices.numpy()
			for conll_idx, sent_predicted, sent_gold in zip(conll_indices, pred_values, gold_depths):
				results[language][conll_idx] = {'predicted': sent_predicted, 'gold': sent_gold}

	return results
