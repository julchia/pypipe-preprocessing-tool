from abc import abstractmethod
from typing import Dict, Any

from omegaconf import OmegaConf

from src.core.interfaces import IProcessBuilder, IProcessHandler, IPipelineProcess


class PipelineProcess(IPipelineProcess):
    """
    """
    
    def __init__(
        self,
        pipeline_conf: OmegaConf,
        process_handlers: Dict[str, IProcessHandler],
        process_builder: IProcessBuilder
    ) -> None:
        self._pipeline_conf = pipeline_conf
        self._process_handlers = process_handlers
        self._process_builder = process_builder
    
    @abstractmethod  
    def _build_process_sequence(self) -> IProcessHandler:
        ...
    
    @abstractmethod
    def get_process(self) -> IProcessHandler:
        ...
    
    def add_step_to_sequence(
        self, 
        handler_name: str, 
        handler_configs: Dict[str, Any]
    ) -> None:
        next_sequence_step = self._process_handlers.get(handler_name)
        self._process_builder._set_next(
            configs=handler_configs, 
            next_step=next_sequence_step
        )