import os
import argparse

from datasets import DependencyDataset
from network import DistanceProbe, DepthProbe
from reporter import DistanceReporter, DepthReporter

import constants

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--train-data", nargs='*', type=str, required=True, help="Conllu files for training")
	parser.add_argument("--train-languages", nargs='*', type=str, required=True, help="Languages of trianing conllu files")
	parser.add_argument("--dev-data", nargs='*', type=str, required=True, help="Conllu files for validation")
	parser.add_argument("--dev-languages", nargs='*', type=str, required=True, help="Languages of validation conllu files")
	parser.add_argument("--test-data", nargs='*', type=str, required=True, help="Conllu files for validation")
	parser.add_argument("--test-languages", nargs='*', type=str, required=True, help="Languages of validation conllu files")
	# Probe arguments
	parser.add_argument("--task", default="distance", type=str, help="Probing task (distance or depth)")
	parser.add_argument("--bert-dim", default=768, type=int, help="Dimensionality of BERT embeddings")
	parser.add_argument("--probe-rank", default=768, type=int, help="Rank of the probe")
	parser.add_argument("--layer_index", default=6, type=int, help="Index of BERT's layer to probe")
	# Train arguments
	parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
	parser.add_argument("--epochs", default=10, type=int, help="Maximal number of training epochs")
	# Specify Bert Model
	parser.add_argument("--bert-dir", default="/net/projects/bert/models/", type=str,
	                    help="Directory where BERT resources are storred (vocab, trained checkpoints)")
	parser.add_argument("--casing", default=constants.CASING_UNCASED, help="Bert model casing")
	parser.add_argument("--language", default=constants.LANGUAGE_MULTILINGUAL, help="Bert model language")
	parser.add_argument("--size", default=constants.SIZE_BASE, help="Bert model size")
	# Reporting options
	parser.add_argument("--report", "-r", action="store_true", help="Whether to report the restults")
	
	
	# parser.add_argument("--threads", default=4, type=int, help="Threads to use")
	args = parser.parse_args()
	
	assert set(args.train_languages) >= set(args.dev_languages),\
		"Currently, evaluation is possible only for languages on which probes were trained"
	assert set(args.train_languages) >= set(args.test_languages),\
		"Currently, evaluation is possible only for languages on which probes were trained"
	assert len(args.train_languages) == len(args.train_data), \
		"Number of train data files and languages needs to be the same"
	assert len(args.dev_languages) == len(args.dev_data), \
		"Number of development data files and languages needs to be the same"
	assert len(args.test_languages) == len(args.test_data), \
		"Number of test data files and languages needs to be the same"
	
	dataset_files = {'train': args.train_data,
	                 'dev': args.dev_data,
	                 'test': args.test_data}
	
	dataset_languages = {'train': args.train_languages,
	                     'dev': args.dev_languages,
	                     'test': args.test_languages}
	
	args.bert_dir = os.path.join(args.bert_dir, "{}-{}-{}".format(args.language, args.size, args.casing))
	if not os.path.exists(args.bert_dir):
		raise ValueError(
			"The requested Bert model combination {}-{}-{} does not exist".format(args.language, args.size,
			                                                                      args.casing))
	
	do_lower_case = (args.casing == "uncased")
	
	dep_dataset = DependencyDataset(dataset_files, dataset_languages, args.task, args.bert_dir, do_lower_case)
	
	if args.task.lower() == 'distance':
		prober = DistanceProbe(args)
	elif args.task.lower() == 'depth':
		prober = DepthProbe(args)
	else:
		raise ValueError("Unknow probing task: {} Choose `depth` or `distance`".format(args.task))
	
	prober.train(dep_dataset,args)
	
	if args.report:
		if args.task.lower() == 'distance':
			reporter = DistanceReporter(prober)
		elif args.task.lower() == 'depth':
			reporter = DepthReporter(prober)
		else:
			raise ValueError("Unknow probing task: {} Choose `depth` or `distance`".format(args.task))

		reporter.predict(dep_dataset.test, args)