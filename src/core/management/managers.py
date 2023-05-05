"""
The module groups utility classes to manage different processes.
"""
from __future__ import annotations

from types import GeneratorType
from typing import (
    List, 
    Dict, 
    Any, 
    Union, 
    Optional, 
    Callable,
    Iterable, 
    Generator
)

import logging

from src.core import paths
from src.core.processes import utils
from src.core.processes.normalization.norm_utils import (
    punctuaction_handler, 
    lowercase_diacritic_handler
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorpusLazyManager:
    """
    The class provides a lazy manager for reading large corpora 
    efficiently.
    
    For efficiency reasons this class doesn't cache the contents, 
    but rather reads them upon first access, meaning you might 
    experience some delay for initial loading time when accessing 
    corpus elements.
    """
    def __init__(self, corpus: Union[List[str], str]) -> None:
        """
        Returns a CorpusLazyManager object.
        
        args:
            corpus: lists of strings or path to files containing 
                the corpus data.
        """
        self._corpus = corpus
        
    def __iter__(self) -> Generator:
        """
        Once the class object has been created, its elements can 
        be accessed iteratively through the iterator protocol, 
        returned by 'CorpusLazyManager.__iter__'.
        """
        type_handler = {
            list: self._yield_corpus_from_list,
            str: self._yield_corpus_from_file,
            GeneratorType: lambda: self._corpus
        }
        
        if type(self._corpus) not in type_handler:
            raise ValueError(
                f"Invalid input type. Expected list, str or GeneratorType. "
                f"Received {type(self._corpus)}"
            )
            
        generator = type_handler[type(self._corpus)]()
        
        yield from generator
        
    def _yield_corpus_from_list(self) -> Generator:
        """Yields over corpus list."""
        for line in self._corpus:
            yield line
    
    def _yield_corpus_from_file(self) -> Generator:
        """Yields over corpus static file."""
        with open(self._corpus, "r") as f:
            for line in f:
                yield line.strip()


class VocabularyManager:
    """Class to process text and extract vocabulary for mapping 
    by implementing standard int2text-text2int bijection."""
    def __init__(
        self, 
        corpus: Union[List[str], str],
        corpus2sent: bool = False,
        text2idx: Dict[str, int] = None,
        add_unk: bool = True, 
        unk_text: str = "<<UNK>>",
        diacritic: bool = False,
        lower_case: bool = False,
        norm_punct: bool = False,
        corpus_iterator: Callable[[Union[List[str], str]], Iterable] = CorpusLazyManager
        ) -> None:
        """
        Builds a VocabularyManager object.
        
        If 'corpus2sent' is False: variable 'text' is equivalent to 
        one unique token.
        If 'corpus2sent' is True: variable 'text' is equivalent to 
        entire unique sentence.
        
        args:
            corpus: List or path to static file (txt or 
                json) of texts to build the vocabulary from.
            corpus2sent: Whether to treat each sentence as a token. 
                Defaults to False
            text2idx: A dictionary mapping tokens to their index in 
                the vocabulary. Defaults to None.
            add_unk: Whether to add an unknown token to the vocabulary. 
                Defaults to True.
            unk_text: The text to use for the unknown token. Defaults 
                to "<<UNK>>".
            diacritic: Whether to handle diacritics. Defaults to False.
            lower_case: Whether to convert text to lowercase. Defaults 
                to False.
            norm_punct: Whether to to handle punctuation. Defaults 
                to False.
            corpus_iterator: Object in charge of managing the lazy creation 
                of a data generator. Defaults to CorpusLazyManager.
        """
        self._corpus_iterator = corpus_iterator
        
        self._corpus2sent = corpus2sent
        
        self._norm_punct = norm_punct
        self._lower_case = lower_case
        self._diacritic = diacritic
        
        if text2idx is None:
            text2idx = {}
        self._text2idx = text2idx

        self._idx2text = {
            idx: text
            for text, idx in self._text2idx.items()
        }
        
        self._add_unk = add_unk
                
        if unk_text is not None and not isinstance(unk_text, str):
            raise ValueError(
                "Invalid unk token object type. Expected str "
                f"but {type(unk_text)} was received"
            )
        
        self._unk_text = unk_text
        
        self.unk_idx = -1
                
        if unk_text is not None:
            self.unk_idx = self.add_text(unk_text)
        
        self.__init_vocab_from_iterable(
            self._corpus_iterator(corpus)
        )
    
    def __str__(self) -> str:
        return f"<VocabularyManager object (size={len(self)})>"
    
    def __len__(self) -> int:
        """Returns the length of the VocabularyManager object based 
        on the number of unique texts it contains."""
        return len(self._text2idx)
    
    def __iter__(self) -> Generator:
        """
        Allows iteration over the VocabularyManager object. For each 
        unique text in the vocabulary, it yields the text itself 
        or a list of individual tokens if '_corpus2sent' is True.
        """
        for text in self._text2idx.keys():
            if self._corpus2sent:
                text = text.split()
            yield text

    def __init_vocab_from_iterable(
        self, 
        iterable: Iterable[List[str]]
    ) -> None:
        """
        Iterates over the given iterable and adds each unique text 
        to the VocabularyManager object. If the VocabularyManager 
        was initialized with corpus2sent=True, each unique text is 
        split into a list of words before being added to the 
        VocabularyManager.
         
        Args:
            iterable: Iterable with corpus of texts.
        """
        for text in iterable:
            self.add_text(text)

    def add_text(self, text: str) -> None:
        """
        Adds a new text to the vocabulary by checking if the text 
        is already present in the vocabulary, and if not, it assigns 
        a new index to it.
        
        args:
            text: Sentence or token present in the corpus.
        """
        if self._lower_case or self._diacritic:
            text = lowercase_diacritic_handler(text=text)
        
        if self._norm_punct:
            text = punctuaction_handler(text=text, repl="")

        if text in self._text2idx:
            idx = self._text2idx[text]
        else:
            idx = len(self._text2idx)
            self._text2idx[text] = idx
            self._idx2text[idx] = text
    
    def get_idx_by_text(self, text: str) -> int:
        """
        Looks up the integer index corresponding to a given piece of 
        text within this vocabulary.
        
        args:
            text: The piece of text whose index you want to find. 
                Must already exist in the vocabulary; otherwise an 
                'unknown word' value will be returned instead.
        """
        if self._add_unk:
            return self._text2idx.get(text, self.unk_idx)
        else:
            return self._text2idx[text]
        
    def get_text_by_index(self, index: int) -> str:
        """
        Looks up the str text corresponding to a given integer index 
        within this vocabulary.
        
        args:
            index: The integer index whose index you want to find. 
                Must already exist in the vocabulary; otherwise a 
                key error exception will be raised.
        """
        if index not in self._idx2text:
            raise KeyError(f"The index {index} is not in the Vocabulary")
        return self._idx2text[index]


class ModelDataManager:
    """The class provides utility methods for managing data and models."""
    @staticmethod
    def get_default_model_path_from_constants(alias: str, file_name: str) -> str:
        """Given an alias and file name, returns the default path to 
        save the model."""
        try:
            default_model_path = paths.MODEL_DEFAULT_PATHS[alias]
            utils.create_dir_if_not_exists(default_model_path)
            logger.info(
                f"Default model path '{default_model_path}' will be create"
            )
            return default_model_path + file_name
        except KeyError:
            raise ValueError(
                f"The alias {alias} is invalid, could not access default model path"
            )
                
    @staticmethod
    def get_default_vocab_path_from_constants(alias: str, file_name: str) -> str:
        """Given an alias and file name, returns the default path to 
        save the vocabulary."""
        try:
            default_vocab_path = paths.VOCAB_DEFAULT_PATHS[alias]
            utils.create_dir_if_not_exists(default_vocab_path)
            logger.info(
                f"Default vocab path '{default_vocab_path}' will be create"
            )
            return default_vocab_path + file_name
        except KeyError:
            raise ValueError(
                f"The alias {alias} is invalid, could not access default vocab path"
            )
        
    def load_data_from_callable(
        self,
        *callback_args,
        callback_fn_to_load_data: Callable[..., None], 
        path_to_load_data: str
    ) -> Optional[Any]:
        """Given a path to load data, calls a callback function to load the data 
        and returns it."""
        if path_to_load_data is None:
            return
        try:
            data = callback_fn_to_load_data(*callback_args, path_to_load_data)
            logger.info(
                f"Data in dir '{path_to_load_data}' has been successfully loaded"
            )
            return data
        except FileNotFoundError:
            logger.warning(
                f"No data to load in dir '{path_to_load_data}'"
            )
            return
        
    def save_data_from_callable(
        self, 
        *callback_args,
        callback_fn_to_save_data: Callable[..., None], 
        path_to_save_data: str, 
        data_file_name: str,
        alias: str,
        to_save_vocab: bool = False 
    ) -> None:
        """
        Given a path to save data and a callback function to save the data, 
        saves the data.
        If no path is provided, saves the data to the default path based on 
        the alias and file name.
        """
        if to_save_vocab:
            fn_to_get_default_path = self.get_default_vocab_path_from_constants
            logger_msg_word = "vocab"
        else:
            fn_to_get_default_path = self.get_default_model_path_from_constants
            logger_msg_word = "model"
        
        if isinstance(path_to_save_data, str):
            path = path_to_save_data + data_file_name
            try:
                callback_fn_to_save_data(*callback_args, path)
                logger.info(
                    f"{logger_msg_word} alias '{alias}' will be stored "
                    f"in path '{path_to_save_data}{data_file_name}'"
                )
            except FileNotFoundError:
                logger.warning(
                    f"No valid path found in '{path_to_save_data}' to store "
                    f"{logger_msg_word} with alias '{alias}'"
                )
                path = fn_to_get_default_path(
                    alias=alias,
                    file_name=data_file_name
                )
                callback_fn_to_save_data(*callback_args, path)
                logger.info(
                    f"{logger_msg_word} alias '{alias}' will be stored "
                    f"in path '{path_to_save_data}{data_file_name}'"
                )
        else:
            return

