###### Paths to pipeline configurations ######

PREPRO_1_CONFIG_PATH = "pypipe/configs/pipe_configs/preprocessing_1.json"

###### Paths to model resources ######

MODEL_DEFAULT_PATHS = {
    "countvec": "pypipe/data/models/countvec",
    "word2vec": "pypipe/data/models/word2vec",
    "regex_norm": "pypipe/data/models/regex_norm"
}

VOCAB_DEFAULT_PATHS = {
    "countvec": "pypipe/data/corpus/countvec",
    "word2vec": "pypipe/data/corpus/word2vec"
}

###### Config alias ######

CONFIG_ALIAS = {
    "prepro_1": PREPRO_1_CONFIG_PATH
}

###### Process alias ######
# The assigned alias must match the name of the process set in a pipeline 
# configurations.

REGEX_NORMALIZER_ALIAS = "regex_norm"
COUNTVEC_FEATURIZER_ALIAS = "countvec"
WORD2VEC_FEATURIZER_ALIAS = "word2vec"




