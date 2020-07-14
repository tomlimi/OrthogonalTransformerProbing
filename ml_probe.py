import os
import argparse
import json

from datasets import DependencyDataset
from network import DistanceProbe, DepthProbe
from reporter import DistanceReporter, DepthReporter, LexicalDistanceReporter

import constants

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	#
	parser.add_argument("parent_dir", type=str, default="/home/limisiewicz/PycharmProjects/MultilingualTransformerProbing/experiments", help="Parent experiment directory")
	parser.add_argument("--json-data", type=str, default=None, help="JSON with conllu and languages for training")

	parser.add_argument("--train-data", nargs='*', type=str, default=None, help="Conllu files for training")
	parser.add_argument("--train-languages", nargs='*', type=str, default=None, help="Languages of trianing conllu files")
	parser.add_argument("--dev-data", nargs='*', type=str, default=None, help="Conllu files for validation")
	parser.add_argument("--dev-languages", nargs='*', type=str, default=None, help="Languages of validation conllu files")
	parser.add_argument("--test-data", nargs='*', type=str, default=None, help="Conllu files for validation")
	parser.add_argument("--test-languages", nargs='*', type=str, default=None, help="Languages of validation conllu files")
	# Probe arguments
	parser.add_argument("--task", default="distance", type=str, help="Probing task (distance, lex-distance, depth or lex-depth)")
	parser.add_argument("--bert-dim", default=768, type=int, help="Dimensionality of BERT embeddings")
	parser.add_argument("--probe-rank", default=768, type=int, help="Rank of the probe")
	parser.add_argument("--layer-index", default=6, type=int, help="Index of BERT's layer to probe")
	# Train arguments
	parser.add_argument("--seed", default=42, type=int, help="Seed for variable initialisation")
	parser.add_argument("--batch-size", default=20, type=int, help="Batch size")
	parser.add_argument("--epochs", default=40, type=int, help="Maximal number of training epochs")
	parser.add_argument("--learning-rate", default=0.001, type=float, help="Initial learning rate")
	parser.add_argument("--ortho", default=None, type=float, help="Orthogonality reguralization (SRIP) for language map matrices.")
	parser.add_argument("--l2", default=None, type=float, help="L2 reguralization of the weights.")
	parser.add_argument("--clip-norm", default=None, type=float, help="Clip gradient norm to this value")
	# Specify Bert Model
	parser.add_argument("--casing", default=constants.CASING_CASED, help="Bert model casing")
	parser.add_argument("--language", default=constants.LANGUAGE_MULTILINGUAL, help="Bert model language")
	parser.add_argument("--size", default=constants.SIZE_BASE, help="Bert model size")
	# Reporting options
	parser.add_argument("--report", "-r", action="store_true", help="Whether to report the restults")
	parser.add_argument("--no-training", "-n", action="store_true", help="Do not conduct probe training, load saved weights and evaluate")
	parser.add_argument("--no-ml-probe", action="store_false", help="Resign from ml probe (store false)")

	# parser.add_argument("--threads", default=4, type=int, help="Threads to use")
	args = parser.parse_args()
	# compatibility
	args.ml_probe = args.no_ml_probe

	if args.json_data:
		with open(args.json_data, 'r') as data_f:
			data_map = json.load(data_f)
		for data_argument, data_value in data_map.items():
			setattr(args, data_argument, data_value)

	if not (args.train_data and args.train_languages and args.dev_data and args.dev_languages and args.test_data and args.test_languages):
		raise ValueError("Train/dev/test data and languages need to be provided in json or arguments.")

	experiment_name = f"task_{args.task.lower()}-layer_{args.layer_index}-trainl_{'_'.join(args.train_languages)}"
	args.out_dir = os.path.join(args.parent_dir,experiment_name)
	if not os.path.exists(args.out_dir):
		os.mkdir(args.out_dir)
	if not args.no_training:
		with open(os.path.join(args.out_dir, 'args.json'), 'wt') as out_f:
			json.dump(vars(args), out_f)

	if not args.no_training:
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

	args.bert_path = "bert-{}-{}-{}".format(args.size, args.language, args.casing)
	do_lower_case = (args.casing == "uncased")

	dep_dataset = DependencyDataset(dataset_files, dataset_languages, args.task, args.bert_path, do_lower_case)

	if args.task.lower() in ('distance', 'lex-distance'):
		prober = DistanceProbe(args)
	elif args.task.lower() in ('depth', 'lex-depth'):
		prober = DepthProbe(args)
	else:
		raise ValueError("Unknow probing task: {} Choose `depth` or `distance`".format(args.task))

	if not args.no_training:
		prober.train(dep_dataset,args)
	else:
		prober.load(args)
	if args.report:
		if args.task.lower() == 'distance':
			test_reporter = DistanceReporter(prober, dep_dataset.test, 'test')
		elif args.task.lower() == 'depth':
			test_reporter = DepthReporter(prober, dep_dataset.test, 'test')
		elif args.task.lower() == 'lex-distance':
			test_reporter = LexicalDistanceReporter(prober, dep_dataset.test, 'test')
		elif args.task.lower() == 'lex-depth':
			raise NotImplementedError
		else:
			raise ValueError("Unknow probing task: {} Choose `depth` or `distance`".format(args.task))

		test_reporter.predict(args)
		test_reporter.write(args)
