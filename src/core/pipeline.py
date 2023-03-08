from typing import Dict

from omegaconf import OmegaConf

from src.core.interfaces import IProcessBuilder
from src.core.handlers import regex_handlers
from src.core.builders.process_builders import ProcessBuilder


class Pipeline:
    """
    """
    
    def __init__(
        self, 
        pipeline_conf: OmegaConf,
        process_builder: IProcessBuilder = ProcessBuilder()
    ):
        """
        """
        
        self.builder_process = process_builder
        self.pipeline_conf = pipeline_conf
    
    def _build_pipeline_secuence(self):
        """
        """
        
        pipe_step = self._build_regex_normalization_steps()
        return pipe_step
    
    def _build_regex_normalization_steps(
        self, 
        process_handlers: Dict = regex_handlers.__dict__
        ):
        """
        """
        
        for handler, state in self.pipeline_conf.pipeline.regex_normalization.items():

            if state:
                next_step = process_handlers.get(handler)
                self.builder_process._set_next(next_step=next_step)
                
        return self.builder_process._build_process()
        
    def get_pipeline(self):
        """
        """
        
        return self._build_pipeline_secuence()
