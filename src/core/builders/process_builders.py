from src.core.interfaces import IProcessHandler, IProcessBuilder


class ProcessBuilder(IProcessBuilder):
    """
    """
        
    def _set_next(self, next_step: IProcessHandler) -> IProcessBuilder:
        """
        """
                    
        try:
            self.process = next_step(self.process)      
        except AttributeError:
            self.process = next_step()
        
        return self
    
    def _build_process(self) -> IProcessHandler:
        """
        """
        
        return self.process
