from typing import List, Dict, Union, Iterable, Generator, Callable

from src.core.management.managers import CorpusLazyManager
from src.core.processes.normalization.norm_utils import (
    punctuaction_handler, 
    lowercase_diacritic_handler
)


class Vocabulary:
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
        Builds a Vocabulary object.
        
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
        return f"<Vocabulary object (size={len(self)})>"
    
    def __len__(self) -> int:
        """Returns the length of the Vocabulary object based 
        on the number of unique texts it contains."""
        return len(self._text2idx)
    
    def __iter__(self) -> Generator:
        """
        Allows iteration over the Vocabulary object. For each 
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
        to the Vocabulary object. If the Vocabulary was initialized
        with corpus2sent=True, each unique text is split into a list
        of words before being added to the Vocabulary.

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

