from abc import abstractclassmethod, abstractmethod

from omegaconf import OmegaConf

from src.core.interfaces import IProcess


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

    @abstractclassmethod
    def get_isolated_process(cls):
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
        
    def get_vocabulary_factory(cls):
        # TODO return class IVocabulary
        pass