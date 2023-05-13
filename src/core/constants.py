from src.core.pipeline.handlers import TextNormalizerHandler, TextFeaturizerHandler
from src.core.processes.normalization.normalizers import RegexNormalizer
from src.core.processes.featurization.featurizers import CountVecFeaturizer, Word2VecFeaturizer


# The assigned alias must match the name of the process set in the pipeline configurations.
REGEX_NORMALIZER_ALIAS = "regex_norm"
COUNTVEC_FEATURIZER_ALIAS = "countvec"
WORD2VEC_FEATURIZER_ALIAS = "word2vec"


PIPELINE_PROCESS_ALIAS = {
    REGEX_NORMALIZER_ALIAS: (TextNormalizerHandler, RegexNormalizer),
    COUNTVEC_FEATURIZER_ALIAS: (TextFeaturizerHandler, CountVecFeaturizer),
    WORD2VEC_FEATURIZER_ALIAS: (TextFeaturizerHandler, Word2VecFeaturizer)
}

