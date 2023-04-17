from __future__ import annotations
from typing import List

import logging
from omegaconf import OmegaConf
from gensim.models import Word2Vec, KeyedVectors

from src.core import constants
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
    
    @staticmethod
    def _get_default_model_path(file_name: str = "/word2vec.model") -> str:
        """
        """
        utils.create_dir_if_not_exists(constants.WORD2VECT_FEATURIZER_MODEL_DEFAULT_PATH)
        return constants.WORD2VECT_FEATURIZER_MODEL_DEFAULT_PATH + file_name
    
    @staticmethod
    def _get_default_vocab_path(file_name: str = "/updated_vocab.txt") -> str:
        """
        """
        utils.create_dir_if_not_exists(constants.WORD2VECT_FEATURIZER_VOCAB_DEFAULT_PATH)
        return constants.WORD2VECT_FEATURIZER_VOCAB_DEFAULT_PATH + file_name
    
    @staticmethod
    def _get_loaded_featurizer_from(path) -> None:
        """
        """
        if path is None:
            return
        try:
            loaded_vect = Word2Vec.load(
                fname=path
            )
        except FileNotFoundError:
            logger.warning(
                f"No 'Word2VecFeaturizer' model to load in dir: '{path}'"
            )
            return
        logger.info(
            f"The 'Word2VecFeaturizer' model in '{path}' has been "
            "successfully loaded"
        )
        return loaded_vect
    
    def _check_if_trained_featurizer_path_exists(self) -> bool:
        """
        """
        self.featurizer = self._get_loaded_featurizer_from(
            path=self._path_to_trained_model
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
            corpus2sent=True
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

    def train(self, trainset: List[str]) -> None:
        """
        """        
        self._vocab = trainset     
        
        if self.featurizer is not None:
            self._train_loaded_featurizer()
        else:
            if self._check_if_trained_featurizer_path_exists():
                self._train_loaded_featurizer()
            else:
                self._train_featurizer_from_scratch()
                
        self.persist()

    def load(self, path_to_trained_model: str = None) -> None:
        """
        """
        if path_to_trained_model is not None:
            self.featurizer = Word2Vec.load(
                fname=path_to_trained_model
            )
            return
        self._check_if_trained_featurizer_path_exists()
    
    def persist(self) -> None:
        """
        """
        if isinstance(self.path_to_save_model, str):
            try:
                path_to_save_featurizer = self.path_to_save_model + "/word2vec.model"
            except TypeError:
                logger.warning(
                    f"No valid path found in '{self.path_to_save_model}' "
                    "to store the 'Word2VecFeaturizer' trained model"
                )
                path_to_save_featurizer = self._get_default_model_path()
            logger.info(
                f"The 'Word2VecFeaturizer' trained model will be stored " 
                f"in '{path_to_save_featurizer}'"
                )
            self.featurizer.save(path_to_save_featurizer)
        else:
            logger.info(
                f"The 'Word2VecFeaturizer' trained model will not be stored "
                "because 'path_to_save_model' is not an str object type"
            )
        
        if isinstance(self.path_to_save_vocabulary, str):
            try:
                path_to_save_voc = self.path_to_save_vocabulary + "/vocab.json"
            except TypeError:
                logger.warning(
                    f"No valid path found in '{self.path_to_save_vocabulary}' "
                    "to store the vocabulary for 'Word2VecFeaturizer'"
                )
                path_to_save_voc = self._get_default_vocab_path()
            logger.info(
                f"The vocabulary for 'Word2VecFeaturizer' "
                f"will be stored in '{path_to_save_voc}'"
            )
            utils.persist_dict_as_json(
                self.featurizer.wv.key_to_index,
                path_to_save_voc
            )
        else:
            logger.info(
                f"The vocabulary for 'Word2VecFeaturizer' object will not be "
                "stored because 'path_to_save_vocabulary' is not an str object " 
                "type"
            )
                                
    def get_word_vector_object(self) -> KeyedVectors:
        if self.featurizer is None:
            logger.warning(
                "It's impossible to get KeyedVectors object because "
                "there is no trained or loaded model"
            )
            return
        return self.featurizer.wv
    
    def process(self, corpus):
        """
        """
        if self.featurizer is None:
            logger.warning(
                "It's impossible to process the input from 'Word2VectFeaturizer' "
                "because there is no trained model"
            )
            return corpus
        
        # TODO chequear input type para wv
        corpus = self.featurizer.wv(corpus)
        
        return corpus

    