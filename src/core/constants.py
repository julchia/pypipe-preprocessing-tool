from omegaconf import OmegaConf

from src.core.processes.normalization.regex_normalizer import RegexNormalizer
from src.core.processes.featurization.count_vect_featurizer import SklearnCountVectorizer


# config paths
PREPROCESSING_CONFIG_PATH = "src/configs/preprocessing_config.json"


# data paths
MODEL_DEFAULT_PATHS = {
    "count_vect_featurizer": "src/data/models/count_vectorizer",
    "word2vec_featurizer": "src/data/models/word2vec"
}

VOCAB_DEFAULT_PATHS = {
    "count_vect_featurizer": "src/data/models/word2vec",
    "word2vec_featurizer": "src/data/corpus/word2vec"
}


# models from omegaconf
PREPROCESSING_CONFIG = OmegaConf.load(PREPROCESSING_CONFIG_PATH)


# processes
PIPELINE_PROCESS_ALIAS = {
    "regex_normalization": RegexNormalizer,
    "count_vect_featurizer": SklearnCountVectorizer
}

