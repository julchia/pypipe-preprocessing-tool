from __future__ import annotations

import re
from unidecode import unidecode

from src.core.interfaces import IProcessorHandler


class Preprocessor(IProcessorHandler):
    def __init__(self, next_preprocessor: IProcessorHandler = None) -> None:
        self.__next_preprocessor = next_preprocessor
    
    def _handle_process(self, text: str) -> str | IProcessorHandler:
        
        preprocessed_text = self._process(text)
        
        if (self.__next_preprocessor is None):
            return preprocessed_text
        else:
            return self.__next_preprocessor._handle_process(preprocessed_text)
        
    def preprocess_text(self, text: str) -> str:
        return self._handle_process(text)


class Punctuation(Preprocessor):
    def _process(self, text: str) -> str:
        return re.sub(r"\W", " ", text)
    

class Digit(Preprocessor):
    def _process(self, text: str) -> str:
        return re.sub("\d+", "", text)
    
    
class SingleWord(Preprocessor):
    def _process(self, text: str) -> str:
        return re.sub(r"\s+[a-zA-Z]\s+", " ", text)


class Uppercase(Preprocessor):
    def _process(self, text: str) -> str:
        return unidecode(text.lower())
    