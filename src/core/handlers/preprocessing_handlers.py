from __future__ import annotations

import re
from unidecode import unidecode

from src.core.interfaces import IProcessorHandler
from src.core.handlers.utils import SubRegexBuilder


class Preprocessor(IProcessorHandler):
    def __init__(self, next_processor: IProcessorHandler = None) -> None:
        self.__next_processor = next_processor
    
    def _handle_process(self, text: str) -> str | IProcessorHandler:
        
        processed_text = self._process(text)
        
        if (self.__next_processor is None):
            return processed_text
        else:
            return self.__next_processor._handle_process(processed_text)
        
    def preprocess_text(self, text: str) -> str:
        return self._handle_process(text)


class WebLinkProcessor(Preprocessor):
    # TODO: ENT
    def _process(self, text: str) -> str:
        return re.sub(r"(http|https|www\.)\S+|\S+.(\.com|\.ar|\.net|\.org|\.info|\.io|\.gov|\.edu|\.tv)", " ", text)    


class EmailProcessor(Preprocessor):
    # TODO: ENT
    def _process(self, text: str) -> str:
        return re.sub(r"[\w\.-]+@[\w\.-]+(\.[\w]+)+", "", text)


class MentionProcessor(Preprocessor):
    # TODO: ENT
    def _process(self, text: str) -> str:
        return re.sub(r"(@|#)[A-Za-z0-9]+", "", text)

        
class PunctuationProcessor(Preprocessor):
    def _process(self, text: str) -> str:
        return re.sub(r"\W", " ", text)
    

class DigitProcessor(Preprocessor):
    def _process(self, text: str) -> str:
        return re.sub("\d+", "", text)
    
    
class SingleWordProcessor(Preprocessor):
    def _process(self, text: str) -> str:
        return re.sub(r"(?<!\S)[^aeiouy](?!\S)", " ", text)


class UppercaseProcessor(Preprocessor):
    def _process(self, text: str) -> str:
        return unidecode(text.lower())


class DuplicatedLetterProcessor(Preprocessor):
    def _process(self, text: str) -> str:
        return re.sub(r"(?!l|r)(.)\1{1,}", r"\1", text)


class IsolatedConsonantProcessor(Preprocessor):
    def _process(self, text: str) -> str:
        return re.sub(r"(?<=\s)[bcdfghjklmnpqrstvwxyz]{2,}(?=\s)", "", text)


class QProcessor(Preprocessor):
    def _process(self, text: str) -> str:
        return SubRegexBuilder(text)\
            .sub(r"\s?(ke|k|qe|q)\s", " que ")\
            .sub(r"\s?(kie|qie)", " quie")\


class ReProcessor(Preprocessor):
    def _process(self, text: str) -> str:
        return re.sub(r"(?<!\S)re(?!\S)", "muy", text)


class LaughtProcessor(Preprocessor):
    def _process(self, text: str) -> str:
        return SubRegexBuilder(text)\
            .sub(r"((ja|aj|ha){3,})", "ja")\
            .sub(r"((je|ej|he){3,})", "je")\
            .sub(r"((ji|ij){3,})", "ji")\
            .sub(r"((jo|oj){3,})", "jo")\
            .sub(r"((ju|uj){3,})", "ju")
