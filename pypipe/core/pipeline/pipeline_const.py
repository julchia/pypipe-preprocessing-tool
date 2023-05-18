from pypipe.configs import config_const
from pypipe.core.pipeline.handlers import TextNormalizerHandler, TextFeaturizerHandler
from pypipe.core.processes.normalization.normalizers import RegexNormalizer
from pypipe.core.processes.featurization.featurizers import CountVecFeaturizer, Word2VecFeaturizer


PIPELINE_PROCESS_ALIAS = {
    config_const.REGEX_NORMALIZER_ALIAS: (TextNormalizerHandler, RegexNormalizer),
    config_const.COUNTVEC_FEATURIZER_ALIAS: (TextFeaturizerHandler, CountVecFeaturizer),
    config_const.WORD2VEC_FEATURIZER_ALIAS: (TextFeaturizerHandler, Word2VecFeaturizer)
}
