from __future__ import annotations
from typing import Dict

from omegaconf import OmegaConf

from src.core.interfaces import IProcessorHandler, IProcessorBuilder
from src.core.handlers.preprocessing_handlers import ExtraWhiteSpacesProcessor


class ProcessorBuilder(IProcessorBuilder):
    """
    """
        
    def _set_next(self, next_step: IProcessorHandler) -> IProcessorBuilder:
        """
        """
                    
        try:
            self.preprocessor = next_step(self.preprocessor)      
        except AttributeError:
            self.preprocessor = next_step()
        
        return self
    
    def _build_processor(self) -> IProcessorHandler:
        """
        """
        
        return self.preprocessor
        

class ProcessorBuilderDirector:
    """
    """
    
    def __init__(
        self, 
        model_conf: OmegaConf, 
        steps_to_build: Dict,
        builder: IProcessorBuilder = ProcessorBuilder()
    ):
        """
        """
        
        self.builder = builder
        self.model_conf = model_conf
        self.steps_to_build = steps_to_build
    
    def _build_processor_steps(self):
        """
        """

        for step, state in self.model_conf.norm_step.items():
            
            if state:
                next_norm_step = self.steps_to_build.get(step)
                self.builder._set_next(next_step=next_norm_step)
                
        return self.builder._build_processor()
     
    def build_preprocessor(self):
        """
        """
        
        if self.model_conf.extra_whitespaces_cleaner:
            self.builder._set_next(next_step=ExtraWhiteSpacesProcessor)
        
        preprocesor = self._build_processor_steps()
        
        return preprocesor
    