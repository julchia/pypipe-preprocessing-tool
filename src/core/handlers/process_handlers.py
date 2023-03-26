from __future__ import annotations
from abc import abstractmethod
from typing import Any

from omegaconf import OmegaConf

from src.core.interfaces import IProcessHandler


class ProcessHandler(IProcessHandler):
    """
    """
    
    def __init__(
        self, 
        configs: OmegaConf, 
        next_processor: IProcessHandler = None
    ) -> None:
        self._configs = configs
        self._next_processor = next_processor
    
    @abstractmethod
    def process(self, apply_to: Any) -> Any:
        ...
    
    def _handle_process(self, request: Any) -> Any | IProcessHandler:
        
        processed_request = self.process(request)
        
        if (self._next_processor is None):
            return processed_request
        else:
            return self._next_processor._handle_process(processed_request)
        