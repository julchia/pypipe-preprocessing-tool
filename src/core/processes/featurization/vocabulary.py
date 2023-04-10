from typing import List


class Vocabulary:
    """
    """
    
    def __init__(self, token2idx=None, add_unk=True, unk_token="<<UNK>>") -> None:
        """
        """
        
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
        
    def add_token(self, token):
        """
        """

        if token in self._token2idx:
            idx = self._token2idx[token]
        else:
            idx = len(self._token2idx)
            self._token2idx[token] = idx
            self._idx2token[idx] = token
        return idx
    
    def get_idx_by_token(self, token):
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
    

# if __name__ == "__main__":
    
    # from string import punctuation
    
    # corpus = "hola mi nombre es juan y me gusta comer hamburguesas."
    
    # vocabulary = Vocabulary()
    
    # # Si punctuationo no forma parte de la
    # # cadena, segmentar punctuation
    
    # for word in corpus.split(" "):
    #     if word not in punctuation:
    #         vocabulary.add_token(word)
            
    # print(vocabulary.get_tokens_from_object_head())
    # print(vocabulary.get_tokens_from_object_tail())