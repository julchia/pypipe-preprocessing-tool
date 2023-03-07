from omegaconf import OmegaConf
from src.core.constants import PREPROCESSING_PIPELINE_CONFIG_PATH


preprocessing_pipeline = OmegaConf.load(PREPROCESSING_PIPELINE_CONFIG_PATH)
