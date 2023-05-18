from abc import abstractclassmethod, abstractmethod
from typing import List, Union, Iterable

from omegaconf import OmegaConf

from pypipe.core.interfaces import IProcess
from pypipe.core.management.managers import DataStorageManager, DataLazyManager


class TextNormalizer(IProcess):
    """Base class for all normalizers that will be implemented 
    either from pipeline or in an isolated way."""
    
    data_manager = DataStorageManager()
    
    def __init__(
        self, 
        configs: OmegaConf, 
        alias: str = None
        ) -> None:
        """
        Builds a TextNormalizer object.

        Args:
            alias: alias to recognize the normalizer within 
                a pipeline (it is None if the normalizer is not 
                within a pipeline).
                
            config: normalizer configurations.
        
        """
        if alias is not None:
            self._configs = configs.pipeline[alias]
        else:
            self._configs = configs
    
    @abstractclassmethod
    def get_isolated_process(cls) -> IProcess:
        """Returns a TextNormalizer object."""
        ...
    
    @abstractclassmethod 
    def get_default_configs(cls) -> OmegaConf:
        """Returns configurations for TextNormalizer object."""
        ...
        
    @abstractmethod
    def persist(self):
        """Persists normalized corpus."""
        ...
    
    @abstractmethod
    def normalize_text(
        self, 
        corpus: Union[List[str], Iterable]
    ) -> Union[List[str], DataLazyManager]:
        """Normalizes the given corpus to remove unwanted characters 
        or symbols from it."""
        ...