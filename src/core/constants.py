from omegaconf import OmegaConf

from src.core import paths
from src.core.pipeline.handlers import TextNormalizerHandler, TextFeaturizerHandler
from src.core.processes.normalization.normalizers import RegexNormalizer
from src.core.processes.featurization.featurizers import CountVecFeaturizer, Word2VecFeaturizer


PREPROCESSING_CONFIG = OmegaConf.load(paths.PREPROCESSING_CONFIG_PATH)


PIPELINE_PROCESS_ALIAS = {
    "regex_norm": (TextNormalizerHandler, RegexNormalizer),
    "countvec": (TextFeaturizerHandler, CountVecFeaturizer),
    "word2vec": (TextFeaturizerHandler, Word2VecFeaturizer)
}

