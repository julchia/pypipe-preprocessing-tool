from omegaconf import OmegaConf

from src.core import paths
from src.core.processes.normalization.regex_normalizer import RegexNormalizer
from src.core.processes.featurization.count_vect_featurizer import SklearnCountVectorizer


PREPROCESSING_CONFIG = OmegaConf.load(paths.PREPROCESSING_CONFIG_PATH)


PIPELINE_PROCESS_ALIAS = {
    "regex_normalization": RegexNormalizer,
    "count_vect_featurizer": SklearnCountVectorizer
}

