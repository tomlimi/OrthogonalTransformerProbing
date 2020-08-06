import tensorflow as tf
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel

import constants
from dependency import DependencyDistance, DependencyDepth
from lexical import LexicalDistance, LexicalDepth


class Dataset:

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _float_features(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _int64_features(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value.reshape(-1)))
    
    # class LanguageTaskData:
    #     def __init__(self, language, task, dependency_data):
    #         self.language = language
    #         self.task = task
    #         self.dependency_data = dependency_data
    #         # self.target, self.mask = dependency_data.target_and_mask()
    #         # self.roots = dependency_data.roots
    #         # self.uu_relations = dependency_data.unlabeled_unordered_relations
    #         # self.punctuation_mask = dependency_data.punctuation_mask
    #
    #     @staticmethod
    #     def serialize_example(target, mask):
    #         feature = {
    #             'target': Dataset._float_features(target),
    #             'mask': Dataset._float_features(mask)
    #         }
    #
    #         return tf.train.Example(features=tf.train.Features(feature=feature))
    #
    #     @staticmethod
    #     def parse(example):
    #         example = tf.io.parse_single_example(example, {
    #             "target": tf.io.FixedLenFeature([constants.MAX_WORDPIECES], tf.float32),
    #             "mask": tf.io.FixedLenFeature([constants.MAX_WORDPIECES], tf.float32)})
    #
    #         # example["image"] = tf.image.convert_image_dtype(tf.image.decode_jpeg(example["image"], channels=3),
    #         #                                                 tf.float32)
    #         # example["mask"] = tf.image.convert_image_dtype(tf.image.decode_png(example["mask"], channels=1), tf.float32)
    #         return example
    #
    #     def write_tfrecord(self):
    #         filename = f'{self.task}_{self.language}'
    #         with tf.io.TFRecordWriter(filename) as writer:
    #             for target, mask in tqdm(self.dependency_data.target_and_mask(), desc=f"Target vector computation, {self.task}"):
    #                 train_example = self.serialize_example(target, mask)
    #                 writer.write(train_example.SerializeToString())
        
    class EmbeddedData:
        def __init__(self, language, tasks, dependency_datasets, bert_model):
            self.language = language
            self.tasks = tasks
            self.target_mask_generators = zip(*(dataset.target_and_mask() for dataset in dependency_datasets))
            self.dependency_data = dependency_datasets[0]
            self.bert_model = bert_model

        def calc_embeddings(self, wordpieces, segments):
            wordpieces = tf.expand_dims(wordpieces, 0)
            segments = tf.expand_dims(segments, 0)
            max_token_len = tf.constant(constants.MAX_TOKENS, shape=(1,), dtype=tf.int64)

            _, _, bert_hidden = self.bert_model(wordpieces, attention_mask=tf.sign(wordpieces), training=False)
            embeddings = bert_hidden[1:]

            # average wordpieces to obtain word representation
            # cut to max nummber of words in batch, note that batch.max_token_len is a tensor, bu all the values are the same
            embeddings = [tf.map_fn(lambda x: tf.math.unsorted_segment_mean(x[0], x[1], x[2]),
                                          (embeddings, segments, max_token_len), dtype=tf.float32) for embeddings in
                                embeddings]
            return embeddings

        @staticmethod
        def serialize_example(embeddings, token_len, task_target_mask):
            feature = {'num_tokens': Dataset._int64_feature(token_len)}
            feature.update({f'layer_{idx}': Dataset._float_features(layer_embeddings.numpy())
                        for idx, layer_embeddings in enumerate(embeddings)})
            
            for task, (target, mask) in task_target_mask.items():
                feature.update({f'target_{task}': Dataset._float_features(target),
                                f'mask_{task}': Dataset._float_features(mask)})
            
            return tf.train.Example(features=tf.train.Features(feature=feature))
        
        def write_tfrecord(self):
            filename = f'bert_{self.language}'
            all_wordpieces, all_segments, all_token_len = self.dependency_data.training_examples()

            with tf.io.TFRecordWriter(filename) as writer:
                for wordpieces, segments, token_len, task_target_mask in \
                        tqdm(zip(all_wordpieces, all_segments, all_token_len, self.target_mask_generators),
                             desc="Embedding computation"):
                    
                    task_target_mask = {task: (target, mask) for task, (target, mask) in zip(self.tasks, task_target_mask)}
                    embeddings = self.calc_embeddings(wordpieces, segments)
                    train_example = self.serialize_example(embeddings, token_len, task_target_mask)
                    writer.write(train_example.SerializeToString())
            
    class DatasetWriter:
        def __init__(self, datafiles, languages, tasks, tokenizer, bert_model):

            for datafile, language in zip(datafiles, languages):
                dependency_datasets = []
                for task in tasks:
                    if task.lower() == "distance":
                        dependency_datasets.append(DependencyDistance(datafile, tokenizer))
                    elif task.lower() == "depth":
                        dependency_datasets.append(DependencyDepth(datafile, tokenizer))
                    elif task.lower() == "lex-distance":
                        dependency_datasets.append(LexicalDistance(datafile, tokenizer, lang=language))
                    elif task.lower() == 'lex-depth':
                        dependency_datasets.append(LexicalDepth(datafile, tokenizer, lang=language))
                    else:
                        raise ValueError(
                            "Unknow probing task: {} Choose `depth`, `lex-depth`, `distance` or `lex-distance`".format(
                                task))
                    
                embedding_data = Dataset.EmbeddedData(language, tasks, dependency_datasets, bert_model)
                embedding_data.write_tfrecord()

    def __init__(self, dataset_files, dataset_languages, dataset_tasks, bert_path, embedding_path=None, do_lower_case=True, read_tfrecord=False):
        assert dataset_files.keys() == dataset_languages.keys(), "Specify the same name of datasets."
        
        self.tasks = dataset_tasks['train']

        if not read_tfrecord:
            tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=do_lower_case)
            bert_model = TFBertModel.from_pretrained(bert_path, output_hidden_states=True)
            for dataset_name in dataset_files.keys():
                self.DatasetWriter(dataset_files[dataset_name],
                                   dataset_languages[dataset_name],
                                   dataset_tasks[dataset_name],
                                   tokenizer,
                                   bert_model)
                
        else:
            setattr(self, 'embeddings', tf.data.TFRecordDataset(embedding_path))

    def parse_factory(self):
        
        def parse(example):
            features_dict = {"num_tokens": tf.io.FixedLenFeature([1], tf.int64)}
            features_dict.update({f"layer_{idx}": tf.io.FixedLenFeature([constants.MAX_WORDPIECES,
                                                                         constants.SIZE_DIMS[constants.SIZE_BASE]],
                                                                        tf.float32)
                                  for idx in range(constants.SIZE_LAYERS[constants.SIZE_BASE])})
            for task in self.tasks:
                if "depth" in task:
                    features_dict.update(
                        {f'target_{task}': tf.io.FixedLenFeature([constants.MAX_WORDPIECES], tf.float32),
                         f'mask_{task}': tf.io.FixedLenFeature([constants.MAX_WORDPIECES], tf.float32)})
            
                elif "distance" in task:
                    features_dict.update(
                        {f'target_{task}': tf.io.FixedLenFeature([constants.MAX_WORDPIECES, constants.MAX_WORDPIECES],
                                                                 tf.float32),
                         f'mask_{task}': tf.io.FixedLenFeature([constants.MAX_WORDPIECES, constants.MAX_WORDPIECES],
                                                               tf.float32)})
                else:
                    ValueError("Task name not recognized. It needs to contain `depth` or `distance in the name")
        
            example = tf.io.parse_single_example(example, features_dict)
            return example
    
        return parse
