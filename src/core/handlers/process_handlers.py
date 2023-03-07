from __future__ import annotations
from src.core.interfaces import IProcessHandler


class ProcessHandler(IProcessHandler):
    """
    """
    
    def __init__(self, next_processor: IProcessHandler = None) -> None:
        self.__next_processor = next_processor
    
    def _handle_process(self, text: str) -> str | IProcessHandler:
        
        processed_text = self._process(text)
        
        if (self.__next_processor is None):
            return processed_text
        else:
            return self.__next_processor._handle_process(processed_text)
        