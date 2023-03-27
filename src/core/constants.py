from omegaconf import OmegaConf

from src.core.pipes.pipeline_processes import *
from src.core.handlers.regex_handlers import *
from src.core.handlers.featurizer_handlers import *


# config paths
PREPROCESSING_CONFIG_PATH = "src/configs/preprocessing_config.json"


# data paths
COUNT_VECTORIZER_MODEL_DEFAULT_PATH = "src/core/models/count_vectorizer"
COUNT_VECTORIZER_VOCAB_DEFAULT_PATH = "src/core/data/corpus/count_vectorizer"


# models from omegaconf
PREPROCESSING_CONFIG = OmegaConf.load(PREPROCESSING_CONFIG_PATH)


# process
PIPELINE_PROCESSES_ALIAS = {
    "regex_normalization": RegexNormalizationProcess,
    "featurization": FeaturizationProcess
}


# process handlers
PROCESS_HANDLERS_ALIAS = {
    "normalization": {
        "normalize_white_spaces": WhiteSpacesHandler,
        "normalize_punctuation": PunctuationHandler,
        "normalize_diacritic": DiacriticHandler,
        "normalize_lowercase": LowercaseHandler,
        "normalize_duplicated_letter": DuplicatedLetterHandler,
        "normalize_mention": MentionHandler,
        "normalize_url": URLHandler,
        "normalize_email": EmailHandler,
        "normalize_digit": DigitHandler,
        "normalize_single_word": SingleWordHandler,
        "normalize_isolated_consonant": IsolatedConsonantHandler,
        "normalize_q": QHandler,
        "normalize_re": ReHandler,
        "normalize_laught": LaughtHandler
    },
    "featurization": {
        "sklearn_count_vectorizer": SklearnCountVectorizer
    }
}