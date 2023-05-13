from __future__ import annotations
from typing import List, Union, Callable, Iterable, Generator

import logging
from functools import reduce
from omegaconf import OmegaConf

from src.core import constants
from src.core.processes import utils
from src.core.management.managers import CorpusLazyManager
from src.core.processes.normalization.base import TextNormalizer
from src.core.processes.normalization.norm_utils import REGEX_NORMALIZATION_HANDLERS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegexNormalizer(TextNormalizer):
    """
    Text normalizer using regex handlers that are stored
    in the constant 'REGEX_NORMALIZATION_HANDLERS'.
    """
    regex_handlers = REGEX_NORMALIZATION_HANDLERS
    
    def __init__(
        self,
        configs: OmegaConf, 
        alias: str = None
    ) -> None:
        """
        Builds a RegexNormalizer object by taking configurations
        from the configs object.
        
        args:
            config: Normalizer configurations.
            
            alias: Alias to recognize the normalizer within a 
                pipeline (it is None if the normalizer is not 
                within a pipeline).
        """
        super().__init__(
            configs=configs,
            alias=alias
        )
        
        self._alias = constants.REGEX_NORMALIZER_ALIAS if alias is None else alias
        
        self.compile_handlers: List = []
        
        self._data_file_name = "/normcorpus.txt"
        self._path_to_save_normcorpus = self._configs.path_to_save_normcorpus
        
    @classmethod
    def get_isolated_process(
        cls, 
        configs: OmegaConf,
    ) -> RegexNormalizer:
        """Returns a RegexNormalizer object."""
        return cls(
            configs=configs
        )
        
    @classmethod
    def get_default_configs(cls) -> OmegaConf:
        """Returns configurations for a RegexNormalizer object."""
        return OmegaConf.create({
            "path_to_save_normcorpus": None, 
            "handlers": {
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
            }
        })
    
    def _compile_regex_handlers(self) -> None:
        """
        Compiles regular expression handlers and adds them to the 
        list of compilation handlers ('compile_handlers'), assuming
        configuration is active.
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
        Performs an string substitution operation where it uses 
        the provided compile_handlers tuple to apply the regular 
        expression patterns to the input text.

        args:
            text: A string representing the input text to be normalized.
            
            compile_handlers: A tuple containing two elements: 0) function, 
                1) string. The function applies the compiled regular 
                expression to the input text, while the string represents 
                the replacement pattern used during normalization.
        """
        handler, repl = compile_handlers
        return handler(
            text=text,
            repl=repl
        )

    def _normalize_text(self, text):
        """
            Normalizes a given input text by applying a series of regular 
            expression patterns using the '_normalize' function. The method 
            returns the normalized text.

            Args:
                text: A string representing the input text to be normalized.
        """
        return reduce(
            self._normalize, 
            self.compile_handlers, 
            text
        )

    def _standard_normalization(self, corpus: List[str], persist: bool = False) -> List[str]:
        """
        Iterates over a list of strings and normalize each string.
        
        args:
            corpus: List of string to normalize.
        """
        norm_corpus = [self._normalize_text(sent) for sent in corpus]
        if persist:
            self.persist(data=norm_corpus)
        return norm_corpus
    
    def _lazy_normalization(self, corpus: Iterable, persist: bool = False) -> Generator:   
        """
        Iterates over a list of strings and performs a lazy normalization
        by yield each element in the list.
        
        args:
            corpus: Iterable to normalize.
        """
        if persist:
            writer = TextNormalizer.data_manager.get_lazy_file_writer(
                path_to_save_data=self._path_to_save_normcorpus,
                data_file_name=self._data_file_name,
                alias=self._alias
            )
            for sent in corpus:
                normalized_sent = self._normalize_text(sent)
                if writer is not None: writer.send(normalized_sent)
                yield normalized_sent
        else:
            for sent in corpus:
                yield self._normalize_text(sent)

    def add_regex_handler(
        self, 
        handler: Callable[[str, str], str], 
        repl: str = None
    ) -> None:
        """
        Adds a custom regular expression handler to the list of compilation 
        handlers ('compile_handlers').
        
        args:
            handler: Callable that takes as arguments: 0) a string to normalize 
                by substitution, 1) substitution element; and should return a 
                string.

            repl: Substitution element
            
            e.g.:
                def regex_norm_handler(text: str, repl: str) -> str:    
                    return re.sub(pattern, repl, text)
        """
        self.compile_handlers.append((handler, repl))
    
    def persist(self, data: Iterable):
        TextNormalizer.data_manager.save_data_from_callable(
            data,
            callback_fn_to_save_data=utils.persist_iterable_as_txtfile,
            path_to_save_data=self._path_to_save_normcorpus,
            data_file_name=self._data_file_name,
            alias=self._alias
        )
    
    def normalize_text(
        self, 
        corpus: Union[List[str], Iterable],
        persist: bool = False
    ) -> Union[List[str], CorpusLazyManager]:
        """
        Normalizes the given corpus by compiling regex handlers set on 
        configurations. Returns either a list or CorpusLazyManager object
        depending on whether the given corpus is a list or iterable.
        
        Args:
            corpus: The text to be normalized. Can accept lists or iterables 
                containing string elements.

        Returns:
            Union[List[str], CorpusLazyManager]: The normalized text as a Python 
            list or a manager object with a generator-like interface providing 
            lazily generated normalized texts. If a list of strings is inputted, 
            returns a list; otherwise a manager object.
        """
        if not isinstance(corpus, (Iterable, List)):
            raise ValueError(
                f"Invalid input type. Expected Iterable or List[str]. "
                f"Received {type(corpus)}"
            )
        
        logger.info("'RegexNormalizer' normalization has started")
        
        self._compile_regex_handlers()
                
        if isinstance(corpus, List):
            return self._standard_normalization(corpus=corpus, persist=persist)
        else:
            return CorpusLazyManager(self._lazy_normalization(corpus=corpus, persist=persist))

