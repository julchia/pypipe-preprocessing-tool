from pypipe import settings
from pypipe.core.pipeline.handlers import TextNormalizerHandler, TextFeaturizerHandler
from pypipe.core.processes.normalization.normalizers import RegexNormalizer
from pypipe.core.processes.featurization.featurizers import CountVecFeaturizer, Word2VecFeaturizer


PIPELINE_PROCESS_ALIAS = {
    settings.REGEX_NORMALIZER_ALIAS: (TextNormalizerHandler, RegexNormalizer),
    settings.COUNTVEC_FEATURIZER_ALIAS: (TextFeaturizerHandler, CountVecFeaturizer),
    settings.WORD2VEC_FEATURIZER_ALIAS: (TextFeaturizerHandler, Word2VecFeaturizer)
}
