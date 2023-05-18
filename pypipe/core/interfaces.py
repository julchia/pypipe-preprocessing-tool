from __future__ import annotations
from abc import ABC, abstractclassmethod, abstractmethod

from omegaconf import OmegaConf


class IProcess(ABC):
    """Common interface to all processes that are incorporated
    into a pipeline."""    
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
    
    
class IProcessHandler(ABC):
    """Common interface to all handlers that incorporate 
    a flow given by a Chain of Responsibility pattern."""    
    @abstractmethod
    def process(self):
        """
        """
        ...
    