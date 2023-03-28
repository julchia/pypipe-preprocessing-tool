from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional


class IProcessHandler(ABC):
    """
    """
    
    @abstractmethod
    def process(self, apply_to: Any) -> Any:
        ...
    
    @abstractmethod
    def _handle_process(self, apply_to: Optional[Any]) -> Optional[IProcessHandler]:
        ...


class IProcessBuilder(ABC):
    """
    """
    
    @abstractmethod
    def _set_next(self, next_step: IProcessHandler) -> IProcessBuilder:
        ...
    
    @abstractmethod
    def _build_process(self) -> IProcessHandler:
        ...
    

class IPipelineProcess(ABC):
    """
    """
    
    @abstractmethod  
    def _build_process_sequence(self) -> IProcessHandler:
        ...
    
    @abstractmethod
    def get_process(self) -> IProcessHandler:
        ...