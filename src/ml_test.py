import os
import argparse
import json

from data_support.tfrecord_wrapper import TFRecordReader
from network import DistanceProbe, DepthProbe
from reporting.reporter import DependencyDistanceReporter, DependencyDepthReporter
from reporting.reporter import LexicalDistanceReporter,  LexicalDepthReporter

import constants

#TODO: Reorganize and check reporting!
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	#
	parser.add_argument("parent_dir", type=str, default="../experiments", help="Parent experiment directory")
	parser.add_argument("data_dir", type=str, default='../resources/tf_data', help="Directory where tfrecord files are stored")
	parser.add_argument("--json-data", type=str, default=None, help="JSON with conllu and languages for training")

	parser.add_argument("--languages", nargs='*', default=['en'], type=str,
	                    help="Languages to probe.")
	parser.add_argument("--task", type=str,
	                    help="Probing tasks (distance, lex-distance, depth or lex-depth)")
	# Probe arguments
	parser.add_argument("--probe-rank", default=768, type=int, help="Rank of the probe")
	parser.add_argument("--no-ml-probe", action="store_true", help="Resign from ml probe (store false)")
	parser.add_argument("--layer-index", default=6, type=int, help="Index of BERT's layer to probe")
	# Specify Bert Model
	parser.add_argument("--casing", default=constants.CASING_CASED, help="Bert model casing")
	parser.add_argument("--language", default=constants.LANGUAGE_MULTILINGUAL, help="Bert model language")
	parser.add_argument("--size", default=constants.SIZE_BASE, help="Bert model size")

	args = parser.parse_args()

	args.bert_path = "bert-{}-{}-{}".format(args.size, args.language, args.casing)
	do_lower_case = (args.casing == "uncased")

	tf_reader = TFRecordReader(args.data_dir, args.bert_path)
	tf_reader.read(args.tasks, args.languages)

	if args.json_data:
		with open(args.json_data, 'r') as data_f:
			data_map = json.load(data_f)
		for data_argument, data_value in data_map.items():
			setattr(args, data_argument, data_value)

	if all(task in ('dep-distance', 'lex-distance') for task in args.tasks):
		prober = DistanceProbe(args)
	elif all(task in ('dep-depth', 'lex-depth') for task in args.tasks):
		prober = DepthProbe(args)
	else:
		raise ValueError(
			"Unknow probing task: {} Choose `depth`, `lex-depth`, `distance` or `lex-distance`".format(args.task))

	prober.load(args)

	if 'dep-distance' in args.tasks:
		test_reporter = DependencyDistanceReporter(prober, tf_reader.test, 'test')
	elif 'dep-depth' in args.tasks:
		test_reporter = DependencyDepthReporter(prober, tf_reader.test, 'test')
	elif 'lex-distance' in args.tasks:
		test_reporter = LexicalDistanceReporter(prober, tf_reader.test, 'test')
	elif 'lex-depth' in args.tasks:
		test_reporter = LexicalDepthReporter(prober, tf_reader.test, 'test')
	else:
		raise ValueError(
			"Unknow probing task: {} Choose `depth`, `lex-depth`, `distance` or `lex-distance`".format(args.task))

	test_reporter.predict(args)
	test_reporter.write(args)
