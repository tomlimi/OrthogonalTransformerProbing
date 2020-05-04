import tensorflow as tf

from bert_wrapper import FullTokenizer



model_path = "/net/projects/bert/models/{}-{}-{}".format(language, size, casing)

tokenizer = FullTokenizer(vocab_file="{}/vocab.txt".format(self.model_path), do_lower_case=self._uncased)


def bert_embeddings(self, sentences):
    """Returns pretrained BERT embeddings for all word in sentences

    Returns pretrained BERT embeddings for all words in sentences.
    The embeddings are predicted from the pretrained BERT model and the
    given layers are averaged. If the input word is tokenized to more
    subwords by BERT, all subword embeddings are averaged into one vector
    for each word in input. Optionally, CLS embedding is also returned
    (as the first embedding).

    The TensorFlow computational graph for the Bert model is constructed
    and the checkpoint loaded during each call.

    Arguments:
        batch: batch of sentences, each being an array of strings.
    Outputs:
        A Python list of sentence embeddings, each a numpy array of shape
        [(1 if cls_embedding else 0) + sentence_length, embedding_size]."""

    def normalize_token(token):
        token = convert_to_unicode(token)
        token = self._tokenizer.basic_tokenizer._clean_text(token)
        token = "".join(c for c in token if not c.isspace())
        if self._uncased: token = token.lower()
        token = self._tokenizer.basic_tokenizer._run_strip_accents(token)
        return token

    # Tokenize input data
    InputFeatures = collections.namedtuple("InputFeatures", "unique_id sentence subwords token_ids input_ids input_mask input_type_ids")
    features, token_subwords = [], []
    stat_tokens, stat_subwords, stat_subword_unks = 0, 0, 0
    for index, sentence in enumerate(sentences):
        # Tokenize into subwords
        subwords = self._tokenizer.tokenize(" ".join(sentence))

        stat_tokens += len(sentence)
        stat_subwords += len(subwords)
        stat_subword_unks += len([subword for subword in subwords if subword.startswith("[UNK]")])

        # Align with original tokens
        token_subwords.append(np.zeros(len(sentence)))
        token_ids, subwords_str, current_token, current_token_normalized = [-1] * len(subwords), "", 0, None
        for i, subword in enumerate(subwords):
            if subword in ["[CLS]", "[SEP]"]: continue

            while current_token_normalized is None:
                current_token_normalized = normalize_token(sentence[current_token])

                if not current_token_normalized:
                    current_token += 1
                    current_token_normalized = None

            if subword.startswith("[UNK]"):
                unk_length = int(subword[6:])
                subwords[i] = subword[:5]
                subwords_str += current_token_normalized[len(subwords_str):len(subwords_str) + unk_length]
            else:
                subwords_str += subword[2:] if subword.startswith("##") else subword
            assert current_token_normalized.startswith(subwords_str)

            token_ids[i] = current_token
            token_subwords[-1][current_token] += 1
            if current_token_normalized == subwords_str:
                subwords_str = ""
                current_token += 1
                current_token_normalized = None

        assert current_token_normalized is None
        while current_token < len(sentence):
            assert not normalize_token(sentence[current_token])
            current_token += 1
        assert current_token == len(sentence)

        # Split into segments with maximum size
        while subwords:
            segment_size = min(len(subwords), self._MAX_SENTENCE_LEN - 2)
            if segment_size < len(subwords):
                while segment_size > 0 and token_ids[segment_size - 1] == token_ids[segment_size]:
                    segment_size -= 1
                assert segment_size > 0

            input_subwords = []
            input_subwords.append("[CLS]")
            input_subwords.extend(subwords[:segment_size])
            input_subwords.append("[SEP]")
            subwords = subwords[segment_size:]

            input_token_ids = np.array([-1] + token_ids[:segment_size] + [-1], dtype=np.int32)
            token_ids = token_ids[segment_size:]

            input_ids = np.zeros(self._MAX_SENTENCE_LEN, dtype=np.int32)
            input_ids[:len(input_subwords)] = self._tokenizer.convert_tokens_to_ids(input_subwords)

            input_mask = np.zeros(self._MAX_SENTENCE_LEN, dtype=np.int8)
            input_mask[:len(input_subwords)] = 1

            input_type_ids = np.zeros(self._MAX_SENTENCE_LEN, dtype=np.int8)

            features.append(InputFeatures(unique_id=len(features),
                                          sentence=index,
                                          subwords=input_subwords,
                                          token_ids=input_token_ids,
                                          input_ids=input_ids,
                                          input_mask=input_mask,
                                          input_type_ids=input_type_ids))

    print("Tokenized {} tokens into {} subwords ({:.3f} per token) with {} UNKs ({:.3f}%)".format(
        stat_tokens, stat_subwords, stat_subwords / stat_tokens,
        stat_subword_unks, 100 * stat_subword_unks / stat_subwords), file=sys.stderr)

    def input_generator():
        for feature in features:
            yield {"unique_ids": feature.unique_id, "input_ids": feature.input_ids,
                   "input_mask": feature.input_mask, "input_type_ids": feature.input_type_ids}

    def input_fn(params):
        dataset = tf.data.Dataset.from_generator(
            input_generator,
            {"unique_ids": tf.int32, "input_ids": tf.int32, "input_mask": tf.int32, "input_type_ids": tf.int32},
            {"unique_ids": [], "input_ids": [self._MAX_SENTENCE_LEN], "input_mask": [self._MAX_SENTENCE_LEN], "input_type_ids": [self._MAX_SENTENCE_LEN]})
        dataset = dataset.batch(batch_size=self._batch_size, drop_remainder=False)
        return dataset