from omegaconf import OmegaConf

# paths
PREPROCESSING_CONFIG_PATH = "src/configs/preprocessing_config.json"


# models from omegaconf
PREPROCESSING_CONFIG = OmegaConf.load(PREPROCESSING_CONFIG_PATH)