from nltk.corpus import wordnet as wn
# Wordpieces | Tokens limits
MAX_TOKENS = 128
MAX_WORDPIECES = 128

# Conllu indices
CONLLU_ID = 0
CONLLU_ORTH = 1
CONLLU_LEMMA = 2
CONLLU_POS = 3
CONLLU_FEATS = 5
CONLLU_HEAD = 6
CONLLU_LABEL = 7

# WordNet constants
# we only consider NOUNs and VERBs, because there is no graph structure for other POS
pos2wnpos = {"NOUN": wn.NOUN,
             "VERB": wn.VERB}
             # "ADJ": wn.ADJ,
             # "ADV": wn.ADV}
             
lang2iso = {'en': 'eng', 'es': 'spa', 'fi': 'fin', 'pl': 'pol', 'ar': 'arb', 'id': 'ind', 'zh': 'cmn', 'fr': 'fra'}

# BERT model parameters
LANGUAGE_ENGLISH = "english"
LANGUAGE_CHINESE = "chinese"
LANGUAGE_MULTILINGUAL = "multilingual"

SIZE_BASE = "base"
SIZE_LARGE = "large"

CASING_CASED = "cased"
CASING_UNCASED = "uncased"

SIZE_DIMS = {"base": 768,
             "large": 1024}

BERT_MODEL_DIR = "/net/projects/bert/models/"