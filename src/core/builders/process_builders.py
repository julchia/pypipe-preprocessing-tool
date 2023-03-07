from __future__ import annotations
from typing import Dict

from omegaconf import OmegaConf

from src.core.interfaces import IProcessHandler, IProcessBuilder
from src.core.handlers import regex_handlers


class ProcessBuilder(IProcessBuilder):
    """
    """
        
    def _set_next(self, next_step: IProcessHandler) -> IProcessBuilder:
        """
        """
                    
        try:
            self.preprocessor = next_step(self.preprocessor)      
        except AttributeError:
            self.preprocessor = next_step()
        
        return self
    
    def _build_processor(self) -> IProcessHandler:
        """
        """
        
        return self.preprocessor
        

class ProcessorBuilderDirector:
    """
    """
    
    def __init__(
        self, 
        pipeline: OmegaConf,
        process_handlers: Dict = regex_handlers.__dict__,
        builder: IProcessBuilder = ProcessBuilder()
    ):
        """
        """
        
        self.builder = builder
        self.process_handlers = process_handlers
        self.pipeline = pipeline
    
    def _build_processor_steps(self):
        """
        """

        for handler, state in self.pipeline.regex_normalization.items():
            
            if state:
                next_norm_step = self.process_handlers.get(handler)
                self.builder._set_next(next_step=next_norm_step)
                
        return self.builder._build_processor()
     
    def build_preprocessor(self):
        """
        """
            
        preprocesor = self._build_processor_steps()
        
        return preprocesor
    