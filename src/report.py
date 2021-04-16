import os
import argparse
import json

from data_support.conll_wrapper import ConllWrapper
from data_support.tfrecord_wrapper import TFRecordReader, TFRecordWriter
from network import Network

from reporting.reporter import CorrelationReporter
from reporting.reporter import SelectedDimensionalityReporter
from reporting.reporter import UASReporter, DependencyDepthReporter

from transformers import BertTokenizer

import constants

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	#
	parser.add_argument("parent_dir", type=str, default="../experiments", help="Parent experiment directory")
	parser.add_argument("data_dir", type=str, default='../resources/tf_data', help="Directory where tfrecord files are stored")
	parser.add_argument("--json-data", type=str, default=None, help="JSON with conllu and languages for training")

	parser.add_argument("--languages", nargs='*', default=['en'], type=str,
	                    help="Languages to probe.")
	parser.add_argument("--tasks", type=str, nargs='*',
	                    help="Probing tasks (distance, lex-distance, depth or lex-depth)")
	# Reporter arguments
	parser.add_argument("--probe-threshold", default=None, type=float,
	                    help="Threshold to filter out dimensions with small values in dependency|depth probe")
	parser.add_argument("--drop-parts", default=None, type=int,
	                    help="When not none the non-zero dimesnions of dependency|depth probe are divided into "
	                         "parts and inference is run multiple times with part of dimension zeroed each time."
	                         "Averaged results are reported (similart to Cross Validation).")

	# Probe arguments
	parser.add_argument("--probe-rank", default=None, type=int, help="Rank of the probe")
	parser.add_argument("--no-ortho-probe", action="store_true", help="Resign from ortho probe (store false)")
	parser.add_argument("--layer-index", default=6, type=int, help="Index of BERT's layer to probe")
	# Specify Transformer Model
	parser.add_argument("--model",
	                    default=f"bert-{constants.SIZE_BASE}-{constants.LANGUAGE_MULTILINGUAL}-{constants.CASING_CASED}",
	                    help="Transformer model name (see: https://huggingface.co/transformers/pretrained_models.html)")

	# Train arguments
	parser.add_argument("--seed", default=42, type=int, help="Seed for variable initialisation")
	parser.add_argument("--batch-size", default=20, type=int, help="Batch size")
	parser.add_argument("--epochs", default=40, type=int, help="Maximal number of training epochs")
	parser.add_argument("--learning-rate", default=0.001, type=float, help="Initial learning rate")
	parser.add_argument("--ortho", default=1., type=float, help="Orthogonality reguralization (SRIP) for language map matrices.")
	parser.add_argument("--l1", default=None, type=float, help="L1 reguralization of the weights.")
	parser.add_argument("--clip-norm", default=None, type=float, help="Clip gradient norm to this value")

	parser.add_argument("--subsample-train", default=None, type=int, help="Size of subsample taken from a training set.")
	parser.add_argument("--zs-dep-languages", nargs='*', default=[], type=str, help="List of languages to disregard in dependency probing training (to evaluate 0 shot capability).")
	parser.add_argument("--fs-dep-languages", nargs='*', default=[], type=str, help="")
	args = parser.parse_args()

	args.ml_probe = not args.no_ortho_probe
	if not args.probe_rank:
		args.probe_rank = constants.MODEL_DIMS[args.model]
		
	do_lower_case = (constants.CASING_UNCASED in args.model)

	if args.seed == 42:
		experiment_name = f"task_{'_'.join(args.tasks)}-layer_{args.layer_index}-trainl_{'_'.join(args.languages)}"
	else:
		experiment_name = f"task_{'_'.join(args.tasks)}-layer_{args.layer_index}-trainl_{'_'.join(args.languages)}-seed_{args.seed}"
	args.out_dir = os.path.join(args.parent_dir,experiment_name)
	if not os.path.exists(args.out_dir):
		os.mkdir(args.out_dir)

	tf_reader = TFRecordReader(args.data_dir, args.model)
	tf_reader.read(args.tasks, args.languages)

	network = Network(args)
	network.load(args)

	if args.probe_threshold and args.drop_parts is None and args.ml_probe:
		# dataset is not required to report dimensionality
		dim_reporter = SelectedDimensionalityReporter(args, network, args.tasks, None, None)
		dim_reporter.compute(args)
		dim_reporter.write(args)
	tasks = set(args.tasks).difference({"rnd_depth", "rnd_distance"})
	if tasks:
		reporter = CorrelationReporter(args, network, tasks, tf_reader.test, 'test')
		reporter.compute(args)
		reporter.write(args)

	# random structure is a control task, so evaluation is done on train set to measure memorization
	rnd_tasks = set(args.tasks).intersection({"rnd_depth", "rnd_distance"})
	if rnd_tasks:
		rnd_reporter = CorrelationReporter(args, network, rnd_tasks, tf_reader.train, 'train')
		rnd_reporter.compute(args)
		rnd_reporter.write(args)

	if 'dep_distance' in args.tasks:
		_, tokenizer = TFRecordWriter.get_model_tokenizer(args.model, do_lower_case=do_lower_case)
		#tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=do_lower_case)
		conll_dict = {}
		for lang in args.languages:
			lang_conll = ConllWrapper(tf_reader.map_conll['test'][args.model][lang]['dep_distance'], tokenizer)
			lang_conll.training_examples()
			conll_dict[lang] = lang_conll

		if 'dep_depth' in args.tasks:
			dep_reportet = DependencyDepthReporter(args, network, tf_reader.test, 'test')
			depths = dep_reportet.compute(args)
		else:
			depths = None
		uas_reporter = UASReporter(args, network, tf_reader.test, 'test', conll_dict, depths)
		uas_reporter.compute(args)
		uas_reporter.write(args)
