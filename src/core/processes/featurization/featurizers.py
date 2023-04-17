from __future__ import annotations
from abc import abstractclassmethod, abstractmethod
from typing import List, Dict, Union

import logging
from omegaconf import OmegaConf

from src.core import constants
from src.core.processes import utils
from src.core.interfaces import IProcess
from src.core.processes.featurization.vocabulary import Vocabulary


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextFeaturizer(IProcess):
    """
    """
        
    def __init__(
        self, 
        alias: str,
        configs: OmegaConf
    ) -> None:
               
        if alias is not None:
            self._configs = configs.pipeline[alias]
        else:
            self._configs = configs
    
    @staticmethod    
    def create_vocab(
        corpus: Union[List[str], str],
        corpus2sent: bool = False,
        text2idx: Dict[str, int] = None, 
        add_unk: bool = True, 
        unk_text: str = "<<UNK>>",
        diacritic: bool = False,
        lower_case: bool = False,
        norm_punct: bool = False
    ) -> Vocabulary:
        return Vocabulary(
            corpus,
            corpus2sent,
            text2idx,
            add_unk,
            unk_text,
            diacritic,
            lower_case,
            norm_punct
        )

    @staticmethod
    def get_default_model_path_from_constants(alias: str, file_name: str) -> str:
        try:
            default_model_path = constants.MODEL_DEFAULT_PATHS[alias]
            utils.create_dir_if_not_exists(default_model_path)
            logger.info(
                f"Default model path '{default_model_path}' will be create"
            )
            return default_model_path + file_name
        except KeyError:
            raise ValueError(
                f"The alias {alias} is invalid, could not access default model path"
            )
                
    @staticmethod
    def get_default_vocab_path_from_constants(alias: str, file_name: str) -> str:
        try:
            default_vocab_path = constants.VOCAB_DEFAULT_PATHS[alias]
            utils.create_dir_if_not_exists(default_vocab_path)
            logger.info(
                f"Default vocab path '{default_vocab_path}' will be create"
            )
            return default_vocab_path + file_name
        except KeyError:
            raise ValueError(
                f"The alias {alias} is invalid, could not access default vocab path"
            )

    @abstractclassmethod
    def get_isolated_process(cls) -> TextFeaturizer:
        """
        """
        ...
    
    @abstractclassmethod 
    def get_default_configs(cls) -> OmegaConf:
        """
        """
        ...
    
    @abstractmethod
    def train(self):
        """
        """
        ...
    
    @abstractmethod
    def load(self):
        """
        """
        ...
        
    @abstractmethod
    def process(self):
        """
        """
        ...
        