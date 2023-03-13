from omegaconf import OmegaConf
from src.core import constants


preprocessing_conf = OmegaConf.load(constants.PREPROCESSING_CONFIG_PATH)
    