import argparse

import constants
from tfrecord_dataset import Dataset

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--train-data", nargs='*', type=str, default=None, help="Conllu files for training")
	parser.add_argument("--train-languages", nargs='*', type=str, default=None, help="Languages of trianing conllu files")
	parser.add_argument("--train-tasks", nargs='*', type=str, default=None, help="Tasks of training conllu files")
	# Specify Bert Model
	parser.add_argument("--casing", default=constants.CASING_CASED, help="Bert model casing")
	parser.add_argument("--language", default=constants.LANGUAGE_MULTILINGUAL, help="Bert model language")
	parser.add_argument("--size", default=constants.SIZE_BASE, help="Bert model size")
	args = parser.parse_args()

	args.bert_path = "bert-{}-{}-{}".format(args.size, args.language, args.casing)
	do_lower_case = (args.casing == constants.CASING_UNCASED)

	dataset_files = {'train': args.train_data}

	dataset_languages = {'train': args.train_languages}

	dataset_tasks = {'train': args.train_tasks}

	Dataset(dataset_files, dataset_languages, dataset_tasks, args.bert_path, do_lower_case)
