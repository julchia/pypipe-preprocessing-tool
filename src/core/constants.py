from src.core.handlers.preprocessing_handlers import *


NORM_CONFIG_PATH = "src/configs/normalizer_config.json"

NORM_STEPS = {
    "punctuation": PunctuationProcessor,
    "web_link": WebLinkProcessor,
    "email": EmailProcessor,
    "mention": MentionProcessor,
    "duplicated_letter": DuplicatedLetterProcessor,
    "digit": DigitProcessor,
    "single_word": SingleWordProcessor,
    "upper_case": UppercaseProcessor,
    "isolated_consonant": IsolatedConsonantProcessor,
    "normalize_q": QProcessor,
    "normalize_re": ReProcessor,
    "normalize_laught": LaughtProcessor
}
