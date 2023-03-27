import re
from unidecode import unidecode

from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError

from src.core.interfaces import IProcessHandler
from src.core.handlers import constants
from src.core.handlers.utils import SubRegexBuilder
from src.core.handlers.process_handlers import ProcessHandler


class RegexNormalizer(ProcessHandler):
    """
    """
    
    def normalize_text(self, text: str) -> str:
        return self._handle_process(text)
    
    def __init__(
        self, 
        configs: OmegaConf, 
        next_processor: IProcessHandler = None
        ) -> None:
        
        super().__init__(configs, next_processor)
        
        try:
            if configs.replacement is not None: 
                self._replacement = configs.replacement
        except ConfigAttributeError:
            pass


class WhiteSpacesHandler(RegexNormalizer):    
    def process(self, text: str) -> str:
        return re.sub(
            constants.WHITE_SPACE_REGEX, 
            self._replacement, 
            text
            )


class MentionHandler(RegexNormalizer):
    def process(self, text: str) -> str:
        return re.sub(
            constants.MENTION_REGEX, 
            self._replacement, 
            text
            )   


class URLHandler(RegexNormalizer):
    def process(self, text: str) -> str:
        return re.sub(
            constants.URL_REGEX, 
            self._replacement, 
            text
            )    


class EmailHandler(RegexNormalizer):
    def process(self, text: str) -> str:
        return re.sub(
            constants.EMAIL_REGEX, 
            self._replacement, 
            text
            )

        
class PunctuationHandler(RegexNormalizer):
    def process(self, text: str) -> str:
        return re.sub(
            constants.PUNCTUTATION_REGEX, 
            self._replacement, 
            text
            )
    
    
class DiacriticHandler(RegexNormalizer):    
    def process(self, text: str) -> str:
        return unidecode(text)
    

class DigitHandler(RegexNormalizer):
    def process(self, text: str) -> str:
        return re.sub(
            constants.DIGIT_REGEX, 
            self._replacement, 
            text
            )
    
    
class SingleWordHandler(RegexNormalizer):
    def process(self, text: str) -> str:
        return re.sub(
            constants.SINGLE_WORD_REGEX, 
            self._replacement, 
            text
            )


class LowercaseHandler(RegexNormalizer):
    def process(self, text: str) -> str:
        return re.sub(
            constants.SPECIAL_CHARS_REGEX, 
            lambda x: x.group().lower(), 
            text
            )


class DuplicatedLetterHandler(RegexNormalizer):
    def process(self, text: str) -> str:
        return re.sub(
            constants.DUPLICATED_LETTER_REGEX, 
            r"\1", 
            text
            )


class IsolatedConsonantHandler(RegexNormalizer):
    def process(self, text: str) -> str:
        return re.sub(
            constants.ISOLATED_CONSONANT_REGEX, 
            self._replacement, 
            text
            )


class QHandler(RegexNormalizer):
    def process(self, text: str) -> str:
        return SubRegexBuilder(text)\
            .sub(constants.Q_REGEX["que"], " que ")\
            .sub(constants.Q_REGEX["quie"], " quie")\


class ReHandler(RegexNormalizer):
    def process(self, text: str) -> str:
        return re.sub(
            constants.RE_REGEX, 
            self._replacement, 
            text
            )


class LaughtHandler(RegexNormalizer):
    def process(self, text: str) -> str:
        return SubRegexBuilder(text)\
            .sub(constants.LAUGHT_REGEX["ja"], "ja")\
            .sub(constants.LAUGHT_REGEX["je"], "je")\
            .sub(constants.LAUGHT_REGEX["ji"], "ji")\
            .sub(constants.LAUGHT_REGEX["jo"], "jo")\
            .sub(constants.LAUGHT_REGEX["ju"], "ju")
