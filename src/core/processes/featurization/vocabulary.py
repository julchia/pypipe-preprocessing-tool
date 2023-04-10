from typing import List

from src.core.processes.normalization.norm_utils import punctuaction_handler


class Vocabulary:
    """
    """
    
    def __init__(
        self, 
        token2idx=None, 
        add_unk=True, 
        unk_token="<<UNK>>",
        norm_punct=False
        ) -> None:
        """
        """
        
        self._norm_punct = norm_punct
        
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
            
    def __str__(self) -> str:
        return f"<Vocabulary object (size={len(self)})>"
    
    def __len__(self) -> int:
        return len(self._token2idx)
    
    def norm_punctuation(self, token: str, repl: str = "") -> str:
        return punctuaction_handler(text=token, repl=repl)
    
    def add_token(self, token: str) -> None:
        """
        """
        if self._norm_punct:
            token = self.norm_punctuation(token)

        if token in self._token2idx:
            idx = self._token2idx[token]
        else:
            idx = len(self._token2idx)
            self._token2idx[token] = idx
            self._idx2token[idx] = token
    
    def add_tokens_from_corpus(self, corpus: List[str]) -> None:
        for sent in corpus:
            list(map(vocabulary.add_token, sent.split(" ")))
    
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
    
    def get_tokens_from_object_head(self, show_first: int = 5) -> List:
        return(list(self._token2idx)[0:show_first])
    
    def get_tokens_from_object_tail(self, show_last: int = 5) -> List:
        return(list(self._token2idx)[-show_last:self.__len__()])
    

if __name__ == "__main__":
        
    corpus = [
        "hola mi nombre es juan y a mí gusta comer hamburguesas.", 
        "a mi primo juan le encanta comer hamburguesas!!"
    ]
    
    vocabulary = Vocabulary(norm_punct=True)
    
    vocabulary.add_tokens_from_corpus(corpus)
    
    print(vocabulary._token2idx)
    print(vocabulary.get_tokens_from_object_head())
    print(vocabulary.get_tokens_from_object_tail())