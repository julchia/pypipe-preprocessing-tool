from omegaconf import OmegaConf

from src.core.interfaces import IProcessHandler, IProcessBuilder


class ProcessBuilder(IProcessBuilder):
    """
    """
        
    def _set_next(self, next_step: IProcessHandler, configs: OmegaConf) -> IProcessBuilder:
        """
        """
                    
        try:
            self.process = next_step(configs=configs, next_processor=self.process)      
        except AttributeError:
            self.process = next_step(configs=configs)
        
        return self
    
    def _build_process(self) -> IProcessHandler:
        """
        """
        
        return self.process
