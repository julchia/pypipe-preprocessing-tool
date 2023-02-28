from src.core.handlers.preprocessing_handlers import *


NORM_CONFIG_PATH = "src/configs/normalizer_config.json"

NORM_STEPS = {
    "punctuation": Punctuation,
    "digit": Digit,
    "single_word": SingleWord,
    "upper_case": Uppercase
}
