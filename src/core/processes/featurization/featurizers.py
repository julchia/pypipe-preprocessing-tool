from __future__ import annotations
from abc import abstractclassmethod, abstractmethod
from typing import List, Dict, Union, Callable

import logging
from omegaconf import OmegaConf

from src.core import paths
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
            default_model_path = paths.MODEL_DEFAULT_PATHS[alias]
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
            default_vocab_path = paths.VOCAB_DEFAULT_PATHS[alias]
            utils.create_dir_if_not_exists(default_vocab_path)
            logger.info(
                f"Default vocab path '{default_vocab_path}' will be create"
            )
            return default_vocab_path + file_name
        except KeyError:
            raise ValueError(
                f"The alias {alias} is invalid, could not access default vocab path"
            )

    def save_data(
        self, 
        *callback_args,
        callback_fn_to_save_data: Callable[..., None], 
        path_to_save_data: str, 
        data_file_name: str,
        alias: str,
        to_save_vocab: bool = False 
    ) -> None:
        """
        """
        if to_save_vocab:
            fn_to_get_default_path = self.get_default_vocab_path_from_constants
            logger_msg_word = "vocab"
        else:
            fn_to_get_default_path = self.get_default_model_path_from_constants
            logger_msg_word = "model"
        
        if isinstance(path_to_save_data, str):
            path = path_to_save_data + data_file_name
            try:
                callback_fn_to_save_data(*callback_args, path)
            except FileNotFoundError:
                logger.warning(
                    f"No valid path found in '{path_to_save_data}' to store "
                    f"{logger_msg_word} with alias '{alias}'"
                )
                path = fn_to_get_default_path(
                    alias=alias,
                    file_name=data_file_name
                )
                callback_fn_to_save_data(*callback_args, path)
        else:
            logger.info(
                f"The {logger_msg_word} with alias '{alias}' will not be stored "
                f"because path '{path_to_save_data}' is not an str object type"
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
        