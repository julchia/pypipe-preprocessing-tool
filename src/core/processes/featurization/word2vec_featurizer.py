from __future__ import annotations
from typing import List, Union, Optional

import logging
from omegaconf import OmegaConf
from gensim.models import Word2Vec, KeyedVectors

from src.core.processes import utils
from src.core.processes.featurization.featurizers import TextFeaturizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Word2VecFeaturizer(TextFeaturizer):
    """
    """
    
    def __init__(
        self,
        configs: OmegaConf, 
        featurizer: Word2Vec = None,
        alias: str = None
    ) -> None:
        """
        """
        
        super().__init__(
            configs=configs,
            alias=alias
        )
        
        self.featurizer = featurizer
        

        self._unk_token = self._configs.unk_token
        
        self.path_to_save_model = self._configs.path_to_save_model
        self.path_to_save_vocabulary = self._configs.path_to_save_vocabulary
        
        self._path_to_trained_model = self._configs.path_to_get_trained_model
        self._path_to_get_stored_vocabulary = self._configs.path_to_get_stored_vocabulary
        
        self._update_stored_vocabulary = self._configs.update_stored_vocabulary
        
    @classmethod
    def get_isolated_process(
        cls,
        configs: OmegaConf, 
        featurizer: Word2Vec = None,
        alias: str = None
    ) -> Word2VecFeaturizer:
        """
        """
        return cls(
            configs=configs,
            featurizer=featurizer,
            alias=alias
        )
        
    @classmethod
    def get_default_configs(cls) -> OmegaConf:
        """
        """
        return OmegaConf.create({
            "method": "cbow",
            "ignore_freq_higher_than": 1,
            "embeddings_size": 64,
            "window": 5,
            "epochs": 5,
            "seed": None,
            "path_to_save_model": None,
            "path_to_save_vocabulary": None,
            "path_to_get_trained_model": None,
            "path_to_get_stored_vocabulary": None,
            "update_stored_vocabulary": False
        })
            
    def _check_if_trained_featurizer_exists_and_load_it(self) -> bool:
        """
        """
        self.featurizer = TextFeaturizer.data_manager.load_data_from_callable(
            callback_fn_to_load_data=Word2Vec.load,
            path_to_load_data=self._path_to_trained_model
        )
        if self.featurizer is None:
            return False
        return True
        
    def _load_featurizer_params(self) -> None:        
        """
        """
        if self._configs.method == "cbow":
            self._sg = 1
            logger.info(
                "The skip-gram architecture to Word2Vec model was setted"
            )
        elif self._configs.method == "skipgram":
            self._sg = 0
            logger.info(
                "The cbow architecture to Word2Vec model was setted"
            )
        
        self._featurizer_params = {
            "sentences": None,
            "min_count": self._configs.ignore_freq_higher_than,
            "vector_size": self._configs.embeddings_size,
            "window": self._configs.window,
            "seed": self._configs.seed,
            "sg": self._sg,
            "workers": 4, # parallel only works if you have installed cpython
            # "negative": "TODO",
            "ns_exponent": 0.75
        }
    
    def _load_train_params(self) -> None:
        self._train_params = {
            "corpus_iterable": self._vocab,
            "total_examples": self.featurizer.corpus_count,
            "epochs": self._configs.epochs,
            "start_alpha": None,
            "end_alpha": None
        }

    def _prepare_vocabulary(self) -> None:
        """
        """
        self._vocab = super().create_vocab(
            corpus=self._vocab, 
            corpus2sent=True,
            unk_text=self._unk_token
        )
        
        if self.featurizer.wv.key_to_index:
            update = self._update_stored_vocabulary
        else:
            update = False
               
        if self._path_to_get_stored_vocabulary is not None: 
            from json import load
            
            with open(self._path_to_get_stored_vocabulary, 'r') as f:
                stored_vocab = load(f)
                   
            self.featurizer.build_vocab_from_freq(
                word_freq=stored_vocab, 
                update=update
            )  
        else:       
            self.featurizer.build_vocab(
                corpus_iterable=self._vocab,
                update=update
            )
    
    def _create_featurizer(self) -> Word2Vec:
        return Word2Vec(**self._featurizer_params)
    
    def _train(self) -> None:
        self._prepare_vocabulary()
        self._load_train_params()
        logger.info("'Word2VecFeaturizer' training has started")
        self.featurizer.train(**self._train_params)
        logger.info("'Word2VecFeaturizer' training finished")
    
    def _train_loaded_featurizer(self) -> None:
        """
        """        
        self._train()
        
    def _train_featurizer_from_scratch(self) -> None:
        """
        """
        self._load_featurizer_params()
        self.featurizer = self._create_featurizer()
        self._train()

    def train(self, trainset: List[str], persist: bool = False) -> None:
        """
        """
        self._vocab = trainset     
        
        if self.featurizer is None:
            if self._check_if_trained_featurizer_exists_and_load_it():
                self._train_loaded_featurizer()
            else:
                self._train_featurizer_from_scratch()
        else:
            self._train_loaded_featurizer()
        
        if persist:
            self.persist()

    def load(self, path_to_trained_model: str = None) -> None:
        """
        """
        if path_to_trained_model is not None:
            self.featurizer = Word2Vec.load(
                fname=path_to_trained_model
            )
        else:
            self._check_if_trained_featurizer_exists_and_load_it()
    
    def persist(self) -> None:
        """
        """
        # try to save model data
        TextFeaturizer.data_manager.save_data_from_callable(
            callback_fn_to_save_data=self.featurizer.save,
            path_to_save_data=self.path_to_save_model,
            data_file_name="/word2vec.model",
            alias="word2vec_featurizer"
        )
        
        # try to save vocab data
        TextFeaturizer.data_manager.save_data_from_callable(
            self.featurizer.wv.key_to_index,
            "w",
            callback_fn_to_save_data=utils.persist_dict_as_json,
            path_to_save_data=self.path_to_save_vocabulary,
            data_file_name="/vocab.json",
            alias="word2vec_featurizer",
            to_save_vocab=True
        )
                                       
    def get_word_vector_object(self) -> Optional[KeyedVectors]:
        if self.featurizer is None:
            logger.warning(
                "It's impossible to get 'KeyedVectors' from 'Word2VecFeaturizer' "
                "object because there is no trained model"
            )
            return
        return self.featurizer.wv
    
    def get_vector_by_key(self, key: Union[str, List[str]]):
        """
        return <class 'numpy.ndarray'>
        """
        if self.featurizer is None:
            logger.warning(
                "It's impossible to get vector items from 'Word2VecFeaturizer' "
                "object because there is no trained model"
            )
            return
        else:
            return self.featurizer.wv.__getitem__(key)
        
