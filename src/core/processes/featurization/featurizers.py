from __future__ import annotations
from abc import abstractclassmethod, abstractmethod
from typing import List, Dict, Union

import logging
from omegaconf import OmegaConf

from src.core.interfaces import IProcess
from src.core.management.managers import ModelDataManager
from src.core.processes.featurization.vocabulary import Vocabulary


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextFeaturizer(IProcess):
    """
    """
    
    data_manager = ModelDataManager()
    
    def __init__(
        self, 
        alias: str,
        configs: OmegaConf,
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
        