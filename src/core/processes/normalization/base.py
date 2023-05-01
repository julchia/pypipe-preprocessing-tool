from abc import abstractclassmethod, abstractmethod
from typing import List, Union, Iterable

from omegaconf import OmegaConf

from src.core.interfaces import IProcess
from src.core.management.managers import CorpusLazyManager


class TextNormalizer(IProcess):
    """
    """
    def __init__(
        self, 
        configs: OmegaConf, 
        alias: str = None
        ) -> None:
        
        if alias is not None:
            self._configs = configs.pipeline[alias]
        else:
            self._configs = configs
    
    @abstractclassmethod
    def get_isolated_process(cls) -> IProcess:
        ...
    
    @abstractclassmethod 
    def get_default_configs(cls) -> OmegaConf:
        """
        """
        ...
    
    @abstractmethod
    def normalize_text(
        self, 
        corpus: Union[List[str], Iterable]
    ) -> Union[List[str], CorpusLazyManager]:
        """
        """
        ...