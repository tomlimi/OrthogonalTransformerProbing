import argparse

import constants
from data_support.tfrecord_wrapper import TFRecordWriter

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("data_dir", type=str, default='../resources/tf_data',
	                    help="Directory where tfrecord files are stored")
	parser.add_argument("--casing", default=constants.CASING_CASED, help="Bert model casing")
	parser.add_argument("--language", default=constants.LANGUAGE_MULTILINGUAL, help="Bert model language")
	parser.add_argument("--size", default=constants.SIZE_BASE, help="Bert model size")
	args = parser.parse_args()

	args.bert_path = "bert-{}-{}-{}".format(args.size, args.language, args.casing)
	models = [args.bert_path]
	languages = ['en']
	data_spec = [
		('train', 'en', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_distance,rnd_depth,pos_distance,pos_depth', '../resources/entrain.conllu'),
		('dev', 'en', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_depth,pos_distance,pos_depth', '../resources/endev.conllu'),
		('test', 'en', 'dep_distance,dep_depth,lex_distance,lex_depth,rnd_depth,pos_distance,pos_depth', '../resources/entest.conllu')
	]

	tasks = ['dep_distance', 'dep_depth', 'lex_distance', 'lex_depth', 'pos_distance', 'pos_depth']

	tf_writer = TFRecordWriter(models, data_spec, args.data_dir)
	tf_writer.compute_and_save(args.data_dir)
