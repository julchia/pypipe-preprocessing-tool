import os


###### Paths to pipeline configurations ######

PREPRO1_CONFIG_PATH = "configs/prepro1.yml"


###### Config alias ######

CONFIG_ALIAS = {
    "prepro1": PREPRO1_CONFIG_PATH
}


###### Process alias ######

# The assigned alias must match the name of the process set in a pipeline 
# configurations.

REGEX_NORMALIZER_ALIAS = "regex_norm"
COUNTVEC_FEATURIZER_ALIAS = "countvec"
WORD2VEC_FEATURIZER_ALIAS = "word2vec"


###### Paths to model resources ######

MODEL_DEFAULT_PATHS = {
    COUNTVEC_FEATURIZER_ALIAS: "pypipe/data/models/countvec",
    WORD2VEC_FEATURIZER_ALIAS: "pypipe/data/models/word2vec",
    REGEX_NORMALIZER_ALIAS: "pypipe/data/models/regex_norm"
}

VOCAB_DEFAULT_PATHS = {
    COUNTVEC_FEATURIZER_ALIAS: "pypipe/data/corpus/countvec",
    WORD2VEC_FEATURIZER_ALIAS: "pypipe/data/corpus/word2vec"
}


###### Log handling ######

LOG_LEVEL = "WARNING"
os.environ["LOG_LEVEL"] = LOG_LEVEL

