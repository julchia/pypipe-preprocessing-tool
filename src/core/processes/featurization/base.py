from __future__ import annotations

from abc import abstractclassmethod, abstractmethod
from typing import List, Dict, Union

from omegaconf import OmegaConf, DictConfig

from src.core.interfaces import IProcess
from src.core.management.managers import ModelDataManager, VocabularyManager


class TextFeaturizer(IProcess):
    """Base class for all featurizers that will be implemented 
    either from pipeline or in an isolated way"""
    
    data_manager = ModelDataManager()
    
    def __init__(
        self, 
        alias: str,
        configs: OmegaConf,
    ) -> None:
        """
        Builds a TextFeaturizer object.

        Args:
            alias: alias to recognize the featurizer within 
                a pipeline (it is None if the featurizer is not 
                within a pipeline).
                
            config: featurizer configurations.
        
        """
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
    ) -> VocabularyManager:
        """Returns a VocabularyManager type generator"""
        return VocabularyManager(
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
        """Returns non-trained TextFeaturizer object"""
        ...
    
    @abstractclassmethod 
    def get_default_configs(cls) -> DictConfig:
        """Returns configurations for TextFeaturizer object"""
        ...
    
    @abstractmethod
    def train(self):
        """Trains TextFeaturizer object"""
        ...
    
    @abstractmethod
    def load(self):
        """Loads TextFeaturizer object"""
        ...
        
    @abstractmethod
    def persist(self):
        """Persists TextFeaturizer object"""
        ...