import re
from unidecode import unidecode
from functools import partial

from src.core.processes.normalization import constants


class SubRegexBuilder(str):
    """
    It extends the str class to implement a builder pattern 
    that allows the sub function to be applied multiple 
    times.
    """

    def __new__(cls, *args, **kwargs):
        newobj = str.__new__(cls, *args, **kwargs)
        newobj.sub = lambda fro, to: SubRegexBuilder(re.sub(fro, to, newobj))
        return newobj


def regex_norm_handler(
    text: str,
    repl: str,
    pattern: str, 
    ) -> str:
    """
    """
    return re.sub(pattern, repl, text)


whitespaces_handler = partial(
    regex_norm_handler, 
    pattern=constants.WHITE_SPACE_REGEX
)


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


mention_handler = partial(
    regex_norm_handler, 
    pattern=constants.MENTION_REGEX, 
)


digit_handler = partial(
    regex_norm_handler, 
    pattern=constants.DIGIT_REGEX, 
)


single_word_handler = partial(
    regex_norm_handler, 
    pattern=constants.SINGLE_WORD_REGEX, 
)


isolated_consonant_handler = partial(
    regex_norm_handler, 
    pattern=constants.ISOLATED_CONSONANT_REGEX, 
) 


re_handler = partial(
    regex_norm_handler, 
    pattern=constants.RE_REGEX, 
) 


duplicated_letter_handler = partial(
    regex_norm_handler,
    repl=r"\1",
    pattern=constants.DUPLICATED_LETTER_REGEX, 
) 


def lowercase_diacritic_handler(
    text: str,
    repl=None,
    pattern=None, 
    ) -> str:
    return unidecode(text).lower()


def q_handler(
    text: str,
    repl = None,
    pattern=constants.Q_REGEX, 
    ) -> str:
    return SubRegexBuilder(text)\
        .sub(pattern["que"], " que ")\
        .sub(pattern["quie"], " quie")
        

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