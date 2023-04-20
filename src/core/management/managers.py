import os
from typing import Any, Optional, Callable

from src.core import paths
from src.core.processes import utils

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

    
class ModelDataManager:
    """
    """
        
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
        
    def load_data_from_callable(
        self,
        *callback_args,
        callback_fn_to_load_data: Callable[..., None], 
        path_to_load_data: str
    ) -> Optional[Any]:
        if path_to_load_data is None:
            return
        try:
            data = callback_fn_to_load_data(*callback_args, path_to_load_data)
            logger.info(
                f"Data in dir '{path_to_load_data}' has been successfully loaded"
            )
            return data
        except FileNotFoundError:
            logger.warning(
                f"No data to load in dir '{path_to_load_data}'"
            )
            return
        
    def save_data_from_callable(
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
                logger.info(
                    f"{logger_msg_word} alias '{alias}' will be stored "
                    f"in path '{path_to_save_data}{data_file_name}'"
                )
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
                logger.info(
                    f"{logger_msg_word} alias '{alias}' will be stored "
                    f"in path '{path_to_save_data}{data_file_name}'"
                )
        else:
            return

