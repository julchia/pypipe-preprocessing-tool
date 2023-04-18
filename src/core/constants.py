from omegaconf import OmegaConf

from src.core import paths
from src.core.processes.normalization.regex_normalizer import RegexNormalizer
from src.core.processes.featurization.countvec_featurizer import CountVecFeaturizer


PREPROCESSING_CONFIG = OmegaConf.load(paths.PREPROCESSING_CONFIG_PATH)


PIPELINE_PROCESS_ALIAS = {
    "regex_normalization": RegexNormalizer,
    "countvec_featurizer": CountVecFeaturizer
}

