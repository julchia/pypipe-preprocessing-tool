from omegaconf import OmegaConf

# paths
PREPROCESSING_CONFIG_PATH = "src/configs/preprocessing_config.json"
COUNT_VECTORIZER_DEFAULT_PATH = "src/core/models/count_vectorizer"

# models from omegaconf
PREPROCESSING_CONFIG = OmegaConf.load(PREPROCESSING_CONFIG_PATH)