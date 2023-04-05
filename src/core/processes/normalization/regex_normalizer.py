from __future__ import annotations
from typing import List, Callable

from functools import reduce
from omegaconf import OmegaConf

from src.core.processes.normalization.text_normalizer import TextNormalizer
from src.core.processes.normalization.norm_utils import REGEX_NORMALIZATION_HANDLERS


class RegexNormalizer(TextNormalizer):
    """
    """
    regex_handlers = REGEX_NORMALIZATION_HANDLERS
    
    def __init__(
        self,
        configs: OmegaConf, 
        alias: str = None
    ) -> None:
           
        super().__init__(
            configs=configs,
            alias=alias
        )
        
        self.compile_handlers: List = []
        
    @classmethod
    def get_isolated_process(
        cls, 
        configs: OmegaConf,
    ) -> RegexNormalizer:
        """
        """
        return cls(
            configs=configs
        )
        
    @classmethod
    def get_default_configs(cls) -> OmegaConf:
        """
        """
        return OmegaConf.create({
            "normalize_laught": {
                "active": True,
                "replacement": None
            },
            "normalize_re": {
                "active": True,
                "replacement": "muy"
            },
            "normalize_q": {
                "active": True,
                "replacement": None
            },
            "normalize_isolated_consonant": {
                "active": True,
                "replacement": ""
            },
            "normalize_single_word": {
                "active": True,
                "replacement": " "
            },
            "normalize_digit": {
                "active": True,
                "replacement": ""
            },
            "normalize_email": {
                "active": True,
                "replacement": "<<EMAIL>>"
            },
            "normalize_url":  {
                "active": True,
                "replacement": "<<URL>>"
            },
            "normalize_mention": {
                "active": True,
                "replacement": "<<MENTION>>"
            },
            "normalize_duplicated_letter": {
                "active": True,
                "replacement": r"\1"
            },
            "normalize_lowercase_diacritic": {
                "active": True,
                "replacement": None
            },
            "normalize_punctuation": {
                "active": True,
                "replacement": ""
            },
            "normalize_white_spaces": {
                "active": True,
                "replacement": " "
            },
        })
    
    def _compile_regex_handlers(self) -> None:
        """
        """
        for regex_handler, config in self._configs.items():
            if config.active:
                if regex_handler in RegexNormalizer.regex_handlers:
                    handler = RegexNormalizer.regex_handlers.get(regex_handler)
                    self.compile_handlers.append(
                        (handler, config.replacement)
                    )
    
    def _normalize(
        self, 
        text: str, 
        compile_handlers: tuple[Callable[[str, str], str], str]
    ) -> str:
        """
        """
        handler, repl = compile_handlers
        return handler(
            text=text,
            repl=repl
        )
    
    def add_regex_handler(
        self, 
        handler: Callable[[str, str], str], 
        repl: str = None
    ) -> None:
        """
        """
        self.compile_handlers.append((handler, repl))
        
    def normalize_text(self, corpus: List[str]) -> List[str]:
        """
        """
        if not isinstance(corpus, List):
            raise ValueError(
                f"List object type expected, {type(corpus)} object received."
            )
        
        self._compile_regex_handlers()
        
        for i, sent in enumerate(corpus):
            corpus[i] = reduce(
                self._normalize, 
                self.compile_handlers, 
                sent
            )
            
        return corpus
    
    