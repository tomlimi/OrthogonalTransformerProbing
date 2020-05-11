import tensorflow as tf
import numpy as np
from tqdm import tqdm

from metrics import UUAS, RootAcc, Spearman

class Reporter():
	
	def __init__(self):
		pass
	
	
class DistanceReporter(Reporter):
	
	def __init__(self,prober):
		self.prober = prober
	
	def predict(self, dataset, args):
		pass
	
	
class DepthReporter(Reporter):
	
	def __init__(self, prober):
		self.prober = prober
		
		self.root_acc = RootAcc()
		self.sprarman_n = Spearman()
		super().__init__()
	
	def predict(self, dataset, args):
		for language in dataset._languages:
			for batch in tqdm(dataset.evaluate_batches(language, size=args.batch_size), desc="Predicting, {}".format(language)):
				predicted = self.prober.predict_on_batch(batch.wordpieces, batch.segments, batch.token_len, batch.max_token_len, batch.language)
				predicted = [sent_predicted.numpy()[:sent_len] for sent_predicted, sent_len
				             in zip(tf.unstack(predicted), batch.token_len)]
				
				self.batch_root_accuracy(batch.roots, predicted)
	
				gold_depths = [sent_gold.numpy()[:sent_len] for sent_gold, sent_len
				                in zip(tf.unstack(batch.target), batch.token_len)]
				self.batch_spearman(gold_depths, predicted)
				
			print("root acccuracy: ", self.root_acc.result())
			print("spearman macro avg correlation: ", self.sprarman_n.result())
		
	def batch_root_accuracy(self,batch_gold, batch_prediction):
		self.root_acc(batch_gold, [sent_predicted.argmax() + 1 for sent_predicted in batch_prediction])
	
	def batch_spearman(self, batch_gold, batch_prediction):
		self.sprarman_n(batch_gold, batch_prediction)
	