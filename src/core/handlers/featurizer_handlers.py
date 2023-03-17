from abc import ABC, abstractclassmethod, abstractmethod
from typing import List, Dict, Any

from omegaconf import OmegaConf
from sklearn.feature_extraction.text import CountVectorizer

from src.core import constants
from src.core.handlers import utils
from src.core.interfaces import IProcessHandler
from src.core.handlers.process_handlers import ProcessHandler
from src.core.handlers import regex_handlers


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


class TextFeaturizer(ProcessHandler):
    """
    """
    
    def __init__(
        self, configs: OmegaConf, 
        next_processor: IProcessHandler = None, 
        vocabulary: Vocabulary = Vocabulary()
        ) -> None:
        super().__init__(next_processor)
        self._configs = configs
        self.vocabulary = vocabulary
    
    def featurize_text(self):
        print("texto a features")


class SklearnCountVectorizer:
    """
    """
    
    def __init__(self, configs) -> None:
        """
        """
        
        # provisional
        self._configs = configs
        
        self._check_if_trained_vectorizer_exists()
        
        self.vectorizer = None
        
        self._path_to_save_model = self._configs.path_to_save_model 
        self._path_to_save_vocabulary = self._configs.path_to_save_vocabulary

    def _check_if_trained_vectorizer_exists(self) -> None:
        try:
            with open(self._configs.path_to_trained_model, 'r') as model:
                self.vectorizer = model
                print("logger.Log(Se cargó el model alojado en path_to_trained_model)")
        except (TypeError, FileNotFoundError):
            print("logger.Warning(No se encontró ningún CountVectorizer model en path_to_trained_model)")
            return
        
    def _check_if_stored_vocabulary_exists(self) -> None:
        try:
            with open(self._configs.path_to_stored_vocabulary, 'r') as vocab:
                self._vocab = vocab
                print("logger.Log(Se cargó el vocabulary alojado en path_to_stored_vocabulary)")    
        except (TypeError, FileNotFoundError):
            print("logger.Warning('No se encontró ningún vocabulary en path_to_stored_vocabulary')")
            self._check_if_will_use_own_vocabulary()
                
    def _check_if_will_use_own_vocabulary(self) -> None:
        if self._configs.use_own_vocabulary_creator:
            self._vocab = None
            print("logger.Log(Se preparará a CountVectorizer para crear un vocabulary nuevo)") 
        else:
            # self._vocab = "dict class Vocabulary"
            print("logger.Log(Se preparará a Vocabulary para crear un vocabulary nuevo)") 
   
    def _load_vectorizer_params_from_configs(self) -> None:
        self._max_features = self._configs.max_features
        self._min_ngram = self._configs.min_ngram
        self._max_ngram = self._configs.max_ngram
        
        self._check_if_stored_vocabulary_exists()
        
        # if regex_handlers.UppercaseHandler in self.__dict__:
        #     self._lowercase = False
        # else:
        self._lowercase = False
            
        # if self._configs.remove_spanish_stop_words:
        #     self._stop_words = "CONSTANTE_CON_SPANISH_STOP_WORDS"
        # else:
        self._stop_words = None
            
    def _load_vectorizer_params_from_default(self) -> None:
        self._strip_accents = None
        self._analyzer = "word"
        
    def _create_vectorizer(self) -> CountVectorizer:
        return CountVectorizer(
            max_features = self._max_features,
            ngram_range = (self._min_ngram, self._max_ngram),
            vocabulary = self._vocab,
            lowercase = self._lowercase,
            strip_accents = self._strip_accents,
            analyzer = self._analyzer,
            stop_words = self._stop_words
        )

    def _get_vectorizer_vocabulary(self) -> None:
        return self.vectorizer.vocabulary_

    def _add_token_to_vocabulary(self):
        pass

    def _train_vectorizer_from_scratch(self, trainset) -> None:
        self.vectorizer = self._create_vectorizer()
        print("logger.Log(Se inicializó un nuevo CountVectorizer object)")
        print("logger.Log(Comenzará el entrenamiento de CountVectorizer object)")  
        self.vectorizer.fit(trainset)
        print("logger.Log(Finalizó entrenamiento de CountVectorizer object)")  

    def train(self, trainset) -> None:
        self._load_vectorizer_params_from_configs()
        self._load_vectorizer_params_from_default()
        self._train_vectorizer_from_scratch(trainset)
        self.persist()

    def persist(self, use_default: bool = False, path_to_save=None) -> None:
        if use_default:
            path_to_save = constants.COUNT_VECTORIZER_MODEL_DEFAULT_PATH
        else:
            try:
                path_to_save = self._path_to_save_model / "vocabularies.pkl"
            except TypeError:
                print("logger.Warning('No se ha podido almacenar el modelo. El path proporcionado no es correcto')")
                return
        
        utils.persist_data_with_pickle(
            self.vectorizer,
            path_to_save,
            'wb'
        )
        
        # print("logger.Log(Se guardó el vectorizer en self._path_to_save_model)")  
        # if self.path_to_save_vocabulary is not None:
        #     vocab = self._get_vectorizer_vocabulary()
        #     utils.persist_data_with_pickle(
        #         vocab,
        #         self._path_to_save_vocabulary,
        #         'wb'
        #     )
        #     print("logger.Log(Se guardó el vocabulary en self.path_to_save_vocabulary)")  
        
    def process(self, corpus):
        if self.vectorizer is None:
            print("logger.Warning(No se puede procesar porque no hay un vecotorizer entrenado)")
            return corpus

        corpus = self.vectorizer.transform(corpus)
        
        return corpus
            


if __name__ == "__main__":
    
    from src.core.constants import PREPROCESSING_CONFIG
    
    vect_configs = PREPROCESSING_CONFIG.pipeline.featurization.SklearnCountVectorizer
    
    trainset = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
    ]
    
    testset = [
        "This is my name", 
        "I like be the third document"
    ]
    
    vect_test = SklearnCountVectorizer(configs=vect_configs)
    
    vect_test.train(trainset)
    
