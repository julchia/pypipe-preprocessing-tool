from __future__ import annotations
from typing import Dict

from omegaconf import OmegaConf

from src.core.interfaces import IProcessorHandler, IProcessorBuilder


class PreprocessorBuilder(IProcessorBuilder):
        
    def set_next(self, next_step: IProcessorHandler) -> IProcessorBuilder:
                    
        try:
            self.preprocessor = next_step(self.preprocessor)      
        except AttributeError:
            self.preprocessor = next_step()
        
        return self
    
    def build_processor(self) -> IProcessorHandler:
        return self.preprocessor
        

class ProcessorBuilderDirector:
        
    def build(
        builder: IProcessorBuilder, 
        model_conf: OmegaConf, 
        steps_to_build: Dict
    ):

        for step, state in model_conf.norm_step.items():
            
            if state:
                next_norm_step = steps_to_build.get(step)
                builder.set_next(next_step=next_norm_step)
                
        return builder.build_processor()
