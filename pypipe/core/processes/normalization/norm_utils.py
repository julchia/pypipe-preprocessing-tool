from typing import Tuple, Dict

import re
from unidecode import unidecode
from functools import partial

from pypipe.core.processes.normalization import constants


class SubRegexBuilder(str):
    """
    It extends the str class to implement a builder pattern 
    that allows the sub function to be applied multiple 
    times.
    
    args:
        args: Positional arguments passed to the constructor.
        kwargs: Keyword arguments passed to the constructor.
    """
    def __new__(cls, *args: Tuple, **kwargs: Dict):
        newobj = str.__new__(cls, *args, **kwargs)
        newobj.sub = lambda fr_, tn_: SubRegexBuilder(
            re.sub(fr_, tn_, newobj)
        )
        return newobj
    
    
def regex_norm_handler(
    text: str,
    repl: str,
    pattern: str, 
    ) -> str:
    """
    General purpose function that replace occurrences of a 
    certain pattern in the given text string.
    
    This function serves as a base for partially applied 
    functions.
    """
    return re.sub(pattern, repl, text)


# 'mi   nombre  ' -> 'mi nombre'
whitespaces_handler = partial(
    regex_norm_handler, 
    pattern=constants.WHITE_SPACE_REGEX
)


# 'mi... nombre' -> 'mi nombre'
punctuaction_handler = partial(
    regex_norm_handler, 
    pattern=constants.PUNCTUTATION_REGEX
)


url_handler = partial(
    regex_norm_handler, 
    pattern=constants.URL_REGEX, 
)


email_handler = partial(
    regex_norm_handler, 
    pattern=constants.EMAIL_REGEX, 
)


# 'mi @nombre' -> 'mi'
mention_handler = partial(
    regex_norm_handler, 
    pattern=constants.MENTION_REGEX, 
)


# 'mi número 3322' -> 'mi número'
digit_handler = partial(
    regex_norm_handler, 
    pattern=constants.DIGIT_REGEX, 
)


# 'mi s nombre' -> 'mi nombre'
single_word_handler = partial(
    regex_norm_handler, 
    pattern=constants.SINGLE_WORD_REGEX, 
)


# 'mi sbl nombre' -> 'mi nombre'
isolated_consonant_handler = partial(
    regex_norm_handler, 
    pattern=constants.ISOLATED_CONSONANT_REGEX, 
) 


# 're loco' -> 'loco'
re_handler = partial(
    regex_norm_handler, 
    pattern=constants.RE_REGEX, 
) 


# 'muy locoooo' -> 'muy loco'
duplicated_letter_handler = partial(
    regex_norm_handler,
    repl=r"\1",
    pattern=constants.DUPLICATED_LETTER_REGEX, 
) 


# 'MI Nombre' -> 'mi nombre'
def lowercase_diacritic_handler(
    text: str,
    repl=None,
    pattern=None, 
    ) -> str:
    return unidecode(text).lower()


# 'kien es' -> 'quien es'
# 'q pasa' -> 'que pasa'
def q_handler(
    text: str,
    repl = None,
    pattern=constants.Q_REGEX, 
    ) -> str:
    return SubRegexBuilder(text)\
        .sub(pattern["que"], " que ")\
        .sub(pattern["quie"], " quie")
        

# 'jajajajajaja' -> 'jaja'
# 'jaajjajjajaj' -> 'jaja'
def laught_handler(
    text: str,
    repl = None,
    pattern=constants.LAUGHT_REGEX, 
    ) -> str:
    return SubRegexBuilder(text)\
        .sub(pattern["ja"], "ja")\
        .sub(pattern["je"], "je")\
        .sub(pattern["ji"], "ji")\
        .sub(pattern["jo"], "jo")\
        .sub(pattern["ju"], "ju")


REGEX_NORMALIZATION_HANDLERS = {
    "normalize_whitespaces": whitespaces_handler,
    "normalize_punctuation": punctuaction_handler,
    "normalize_lowercase_diacritic": lowercase_diacritic_handler,
    "normalize_duplicated_letter": duplicated_letter_handler,
    "normalize_mention": mention_handler,
    "normalize_url": url_handler,
    "normalize_email": email_handler,
    "normalize_digit": digit_handler,
    "normalize_single_word": single_word_handler,
    "normalize_isolated_consonant": isolated_consonant_handler,
    "normalize_q": q_handler,
    "normalize_re": re_handler,
    "normalize_laught": laught_handler
}