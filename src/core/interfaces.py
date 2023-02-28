from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class IProcessorHandler(ABC):
    
    @abstractmethod
    def _process(self, apply_to: Optional(str)) -> Optional(str):
        pass
    
    @abstractmethod
    def _handle_process(self, apply_to: Optional(str)) -> Optional(IProcessorHandler):
        pass


class IProcessorBuilder(ABC):
    
    @abstractmethod
    def set_next(self, next_step: IProcessorHandler) -> IProcessorBuilder:
        pass
    
    @abstractmethod
    def build_processor(self) -> IProcessorHandler:
        pass