from omegaconf import OmegaConf
from src.core.constants import PREPROCESSING_CONFIG_PATH


preprocessing_conf = OmegaConf.load(PREPROCESSING_CONFIG_PATH)
    