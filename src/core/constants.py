from omegaconf import OmegaConf

from src.core.processes.normalization.regex_normalizer import RegexNormalizer
from src.core.processes.featurization.count_vect_featurizer import SklearnCountVectorizer


# config paths
PREPROCESSING_CONFIG_PATH = "src/configs/preprocessing_config.json"


# data paths
COUNT_VECTORIZER_MODEL_DEFAULT_PATH = "src/data/models/count_vectorizer"
COUNT_VECTORIZER_VOCAB_DEFAULT_PATH = "src/data/corpus/count_vectorizer"


# models from omegaconf
PREPROCESSING_CONFIG = OmegaConf.load(PREPROCESSING_CONFIG_PATH)


# process
PIPELINE_PROCESS_ALIAS = {
    "regex_normalization": RegexNormalizer,
    "sklearn_count_vect": SklearnCountVectorizer
}

