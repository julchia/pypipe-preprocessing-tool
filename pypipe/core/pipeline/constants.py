from pypipe.configs import constants
from pypipe.core.pipeline.handlers import TextNormalizerHandler, TextFeaturizerHandler
from pypipe.core.processes.normalization.normalizers import RegexNormalizer
from pypipe.core.processes.featurization.featurizers import CountVecFeaturizer, Word2VecFeaturizer


PIPELINE_PROCESS_ALIAS = {
    constants.REGEX_NORMALIZER_ALIAS: (TextNormalizerHandler, RegexNormalizer),
    constants.COUNTVEC_FEATURIZER_ALIAS: (TextFeaturizerHandler, CountVecFeaturizer),
    constants.WORD2VEC_FEATURIZER_ALIAS: (TextFeaturizerHandler, Word2VecFeaturizer)
}
