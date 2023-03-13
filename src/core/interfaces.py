from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from omegaconf import OmegaConf


class IProcessHandler(ABC):
    """
    """
    
    next_processor: IProcessHandler
    configs: OmegaConf
        
    @abstractmethod
    def _process(self, apply_to: Optional(str)) -> Optional(str):
        pass
    
    @abstractmethod
    def _handle_process(self, apply_to: Optional(str)) -> Optional(IProcessHandler):
        pass


class IProcessBuilder(ABC):
    """
    """
    
    @abstractmethod
    def _set_next(self, next_step: IProcessHandler) -> IProcessBuilder:
        pass
    
    @abstractmethod
    def _build_process(self) -> IProcessHandler:
        pass