import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os

from metrics import UUAS, RootAcc, Spearman

class Reporter():
	
	def __init__(self, prober, dataset, dataset_name):
		self.prober = prober
		self.dataset = dataset
		self.dataset_name = dataset_name
	
	
class DistanceReporter(Reporter):
	
	def __init__(self,prober):
		self.prober = prober
	
	def predict(self, dataset, args):
		pass
	
	
class DepthReporter(Reporter):
	
	def __init__(self, prober, dataset, dataset_name):
		super().__init__(prober, dataset, dataset_name)
		
		self.root_acc = dict()
		self.sprarman_n = dict()
		
	def write(self, args):
		for language in self.dataset._languages:
			prefix = '{}.{}.'.format(self.dataset_name, language)
			with open(os.path.join(args.out_dir, prefix+'root_acc'), 'w') as root_acc_f:
				root_acc_f.write(str(self.root_acc[language].result()))
			
			with open(os.path.join(args.out_dir, prefix + 'spearman'), 'w') as sperarman_f:
				for sent_l, val in self.sprarman_n[language].result().items():
					sperarman_f.write(f'{sent_l}\t{val}\n')
				
			with open(os.path.join(args.out_dir, prefix + 'spearman_mean'), 'w') as sperarman_mean_f:
				result = str(np.nanmean(np.fromiter(self.sprarman_n[language].result().values(), dtype=float)))
				sperarman_mean_f.write(result)
	
	def predict(self, args):
		for language in self.dataset._languages:
			self.root_acc[language] = RootAcc()
			self.sprarman_n[language] = Spearman()
			for batch in tqdm(self.dataset.evaluate_batches(language, size=args.batch_size), desc="Predicting, {}".format(language)):
				predicted = self.prober.predict_on_batch(batch.wordpieces, batch.segments, batch.token_len, batch.max_token_len, batch.language)
				predicted = [sent_predicted.numpy()[:sent_len] for sent_predicted, sent_len
				             in zip(tf.unstack(predicted), batch.token_len)]
				
				self.batch_root_accuracy(language, batch.roots, predicted)
	
				gold_depths = [sent_gold.numpy()[:sent_len] for sent_gold, sent_len
				                in zip(tf.unstack(batch.target), batch.token_len)]
				self.batch_spearman(language, gold_depths, predicted)
		
	def batch_root_accuracy(self,language,batch_gold, batch_prediction):
		self.root_acc[language](batch_gold, [sent_predicted.argmax() + 1 for sent_predicted in batch_prediction])
	
	def batch_spearman(self, language, batch_gold, batch_prediction):
		self.sprarman_n[language](batch_gold, batch_prediction)
