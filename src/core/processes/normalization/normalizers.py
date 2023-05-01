from __future__ import annotations
from typing import List, Union, Callable, Iterable, Generator

import logging
from functools import reduce
from omegaconf import OmegaConf

from src.core.management.managers import CorpusLazyManager
from src.core.processes.normalization.base import TextNormalizer
from src.core.processes.normalization.norm_utils import REGEX_NORMALIZATION_HANDLERS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        for regex_handler, spec in self._configs.handlers.items():
            if self._configs.active:
                if regex_handler in RegexNormalizer.regex_handlers:
                    handler = RegexNormalizer.regex_handlers.get(regex_handler)
                    self.compile_handlers.append(
                        (handler, spec.replacement)
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

    def _standard_normalization(self, corpus: List[str]) -> List[str]:
        norm_corpus = []
        for sent in corpus:
            norm_corpus.append(
                reduce(
                    self._normalize, 
                    self.compile_handlers, 
                    sent
                )
            )
        return norm_corpus
    
    def _lazy_normalization(self, corpus: Iterable) -> Generator:
        for sent in corpus:
            yield reduce(
                self._normalize, 
                self.compile_handlers, 
                sent
            )

    def add_regex_handler(
        self, 
        handler: Callable[[str, str], str], 
        repl: str = None
    ) -> None:
        """
        """
        self.compile_handlers.append((handler, repl))
    
    def normalize_text(
        self, 
        corpus: Union[List[str], Iterable]
    ) -> Union[List[str], CorpusLazyManager]:
        """
        """
        if not isinstance(corpus, (Iterable, List)):
            raise ValueError(
                f"Invalid input type. Expected Iterable or List[str]. "
                f"Received {type(corpus)}"
            )
        
        logger.info("'RegexNormalizer' normalization has started")
        
        self._compile_regex_handlers()
                
        if isinstance(corpus, List):
            return self._standard_normalization(corpus=corpus)
        else:
            return CorpusLazyManager(self._lazy_normalization(corpus=corpus))

