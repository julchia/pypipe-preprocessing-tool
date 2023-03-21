from omegaconf import OmegaConf


# config paths
PREPROCESSING_CONFIG_PATH = "src/configs/preprocessing_config.json"


# data paths
COUNT_VECTORIZER_MODEL_DEFAULT_PATH = "src/core/models/count_vectorizer/"
COUNT_VECTORIZER_VOCAB_DEFAULT_PATH = "src/core/data/corpus/count_vectorizer/"


# models from omegaconf
PREPROCESSING_CONFIG = OmegaConf.load(PREPROCESSING_CONFIG_PATH)