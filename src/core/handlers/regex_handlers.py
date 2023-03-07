import re
from unidecode import unidecode

from src.core.handlers.process_handlers import ProcessHandler
from src.core.handlers.utils import SubRegexBuilder

    
class RegexNormalizer(ProcessHandler):
    """
    """
    
    def normalize_text(self, text: str) -> str:
        return self._handle_process(text)


class WhiteSpacesHandler(RegexNormalizer):
    def _process(self, text: str) -> str:
        return re.sub(r'\s+', " ", text)    


class WebLinkHandler(RegexNormalizer):
    # TODO: ENT
    def _process(self, text: str) -> str:
        return re.sub(r'(http|https|www\.)\S+|\S+.(\.com|\.ar|\.net|\.org|\.info|\.io|\.gov|\.edu|\.tv)', " ", text)    


class EmailHandler(RegexNormalizer):
    # TODO: ENT
    def _process(self, text: str) -> str:
        return re.sub(r'[\w\.-]+@[\w\.-]+(\.[\w]+)+', "", text)


class MentionHandler(RegexNormalizer):
    # TODO: ENT
    def _process(self, text: str) -> str:
        return re.sub(r'(@|#)[A-Za-z0-9]+', "", text)

        
class PunctuationHandler(RegexNormalizer):
    def _process(self, text: str) -> str:
        return re.sub(r'\W', " ", text)
    
    
class DiacriticHandler(RegexNormalizer):
    def _process(self, text: str) -> str:
        return unidecode(text)
    

class DigitHandler(RegexNormalizer):
    def _process(self, text: str) -> str:
        return re.sub(r'\d+', "", text)
    
    
class SingleWordHandler(RegexNormalizer):
    def _process(self, text: str) -> str:
        return re.sub(r'(?<!\S)[^aeiouy](?!\S)', " ", text)


class UppercaseHandler(RegexNormalizer):
    def _process(self, text: str) -> str:
        return re.sub(r'[a-zA-ZáéíóúÁÉÍÓÚñÑüÜàèìòùÀÈÌÒÙäëïöüÄËÏÖÜâêîôûÂÊÎÔÛ]+', lambda x: x.group().lower(), text)


class DuplicatedLetterHandler(RegexNormalizer):
    def _process(self, text: str) -> str:
        return re.sub(r'(?!l|r)(.)\1{1,}', r"\1", text)


class IsolatedConsonantHandler(RegexNormalizer):
    def _process(self, text: str) -> str:
        return re.sub(r'(?<=\s)[bcdfghjklmnpqrstvwxyz]{2,}(?=\s)', "", text)


class QHandler(RegexNormalizer):
    def _process(self, text: str) -> str:
        return SubRegexBuilder(text)\
            .sub(r'\s?(ke|k|qe|q)\s', " que ")\
            .sub(r'\s?(kie|qie)', " quie")\


class ReHandler(RegexNormalizer):
    def _process(self, text: str) -> str:
        return re.sub(r'(?<!\S)re(?!\S)', "muy", text)


class LaughtHandler(RegexNormalizer):
    def _process(self, text: str) -> str:
        return SubRegexBuilder(text)\
            .sub(r'((ja|aj|ha){3,})', "ja")\
            .sub(r'((je|ej|he){3,})', "je")\
            .sub(r'((ji|ij){3,})', "ji")\
            .sub(r'((jo|oj){3,})', "jo")\
            .sub(r'((ju|uj){3,})', "ju")
