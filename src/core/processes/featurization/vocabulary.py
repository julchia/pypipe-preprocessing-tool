from typing import List, Dict, Iterable, Generator

from src.core.processes.normalization.norm_utils import punctuaction_handler


class Vocabulary:
    """
    """
    
    def __init__(
        self, 
        corpus: List[str] = None,
        token2idx: Dict[str, int] = None, 
        add_unk: bool = True, 
        unk_token: str = "<<UNK>>",
        lower_case: bool = False,
        norm_punct: bool = False,
        ) -> None:
        """
        """
        self._corpus = corpus
        
        self._norm_punct = norm_punct
        self._lower_case = lower_case
        
        if token2idx is None:
            token2idx = {}
        self._token2idx = token2idx

        self._idx2token = {
            idx: token
            for token, idx in self._token2idx.items()
        }
        
        self._add_unk = add_unk
        self._unk_token = unk_token
        
        self.unk_idx = -1
        if add_unk:
            self.unk_idx = self.add_token(unk_token)
    
    def __iter__(self) -> Generator[str]:
        if self._corpus is not None:
            self._init_vocab_from_iterable(self._iter_corpus(self._corpus))
        for token in self._token2idx.keys():
            yield token

    def __str__(self) -> str:
        return f"<Vocabulary object (size={len(self)})>"
    
    def __len__(self) -> int:
        return len(self._token2idx)
    
    def _iter_corpus(self, corpus: List[str]) -> Generator[str]:
        for sent in corpus:
            for word in sent.split(" "):
                yield word
                
    def _init_vocab_from_iterable(self, iterable: Iterable[List[str]]) -> None:
        for token in iterable:
            self.add_token(token)
    
    def add_token(self, token: str) -> None:
        """
        """
        if self._lower_case:
            token = token.lower()
        
        if self._norm_punct:
            token = punctuaction_handler(text=token, repl="")

        if token in self._token2idx:
            idx = self._token2idx[token]
        else:
            idx = len(self._token2idx)
            self._token2idx[token] = idx
            self._idx2token[idx] = token
    
    def get_idx_by_token(self, token: str):
        """
        """
        if self._add_unk:
            return self._token2idx.get(token, self.unk_idx)
        else:
            return self._token2idx[token]
        
    def get_token_by_index(self, index):
        """
        """
        if index not in self._idx2token:
            raise KeyError(f"The index {index} is not in the Vocabulary")
        return self._idx2token[index]
