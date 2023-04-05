from __future__ import annotations
from abc import ABC, abstractclassmethod

from omegaconf import OmegaConf


class IProcess(ABC):
    """
    """
    
    @abstractclassmethod
    def get_isolated_process(cls) -> IProcess:
        """
        """
        ...
    
    @abstractclassmethod 
    def get_default_configs(cls) -> OmegaConf:
        """
        """
        ...