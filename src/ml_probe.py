import os
import argparse
import json

from data_support.tfrecord_wrapper import TFRecordReader
from network import Network

import constants

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	#
	parser.add_argument("parent_dir", type=str, default="../experiments", help="Parent experiment directory")
	parser.add_argument("data_dir", type=str, default='../resources/tf_data', help="Directory where tfrecord files are stored")
	parser.add_argument("--json-data", type=str, default=None, help="JSON with conllu and languages for training")

	parser.add_argument("--languages", nargs='*', default=['en'], type=str,
	                    help="Languages to probe.")
	parser.add_argument("--tasks", nargs='*', type=str,
	                    help="Probing tasks (distance, lex-distance, depth or lex-depth)")

	# Probe arguments
	parser.add_argument("--probe-rank", default=768, type=int, help="Rank of the probe")
	parser.add_argument("--no-ml-probe", action="store_true", help="Resign from ml probe (store false)")
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



	args = parser.parse_args()
	# compatibility
	args.ml_probe = not args.no_ml_probe

	if args.json_data:
		with open(args.json_data, 'r') as data_f:
			data_map = json.load(data_f)
		for data_argument, data_value in data_map.items():
			setattr(args, data_argument, data_value)

	experiment_name = f"task_{'_'.join(args.tasks)}-layer_{args.layer_index}-trainl_{'_'.join(args.languages)}"
	args.out_dir = os.path.join(args.parent_dir,experiment_name)
	if not os.path.exists(args.out_dir):
		os.mkdir(args.out_dir)

	args.bert_path = "bert-{}{}-{}".format(args.size, args.language, args.casing)
	do_lower_case = (args.casing == "uncased")

	tf_reader = TFRecordReader(args.data_dir, args.bert_path)
	tf_reader.read(args.tasks, args.languages)

	network = Network(args)

	network.train(tf_reader,args)
