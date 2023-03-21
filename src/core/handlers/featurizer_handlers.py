import logging
from abc import ABC, abstractclassmethod, abstractmethod
from typing import List, Set, Dict, Any

from omegaconf import OmegaConf
from sklearn.feature_extraction.text import CountVectorizer

from src.core import constants
from src.core.handlers import utils
from src.core.interfaces import IProcessHandler
from src.core.handlers.process_handlers import ProcessHandler
from src.core.handlers import regex_handlers


logger = logging.getLogger(__name__)


class TextFeaturizer(ProcessHandler):
    """
    """
    
    def __init__(
        self, configs: OmegaConf, 
        next_processor: IProcessHandler = None, 
        # vocabulary: Vocabulary = Vocabulary()
        ) -> None:
        super().__init__(next_processor)
        self._configs = configs
        # self.vocabulary = vocabulary
    
    def featurize_text(self):
        print("texto a features")


class SklearnCountVectorizer:
    """
    """
    
    def __init__(
        self, 
        configs: OmegaConf, 
        vectorizer: CountVectorizer = None
    ) -> None:
        """
        """
        
        # provisional
        self._configs = configs
        
        self.vectorizer = vectorizer
        
        self._path_to_trained_model = self._configs.path_to_trained_model
        self._path_to_stored_vocabulary = self._configs.path_to_stored_vocabulary
        
        self.path_to_save_model = self._configs.path_to_save_model 
        self.path_to_save_vocabulary = self._configs.path_to_save_vocabulary
        
        self._update_stored_vocabulary = self._configs.update_stored_vocabulary

    @classmethod
    def get_vectorizer_as_isolated_model(
        cls, 
        configs: OmegaConf, 
        vectorizer: CountVectorizer = None
    ):
        """
        """
        return cls(
            configs=configs,
            vectorizer=vectorizer
        )
        
    @classmethod
    def get_default_configs(cls) -> Dict[str, Any]:
        """
        """
        return {
            "max_features": None,
            "min_ngram": 1,
            "max_ngram": 1,
            "remove_spanish_stop_words": False,
            "path_to_trained_model": "",
            "path_to_stored_vocabulary": "",
            "update_stored_vocabulary": False,
            "use_own_vocabulary_creator": True,
            "unk_token": "<<UNK>>",
            "path_to_save_model": None,
            "path_to_save_vocabulary": None
        }

    # @classmethod
    # def get_vocabulary_factory(cls):
    #     # return class Vocabulary
    #     pass

    @staticmethod
    def _get_default_model_path(file_name: str = "/vocabularies.pkl") -> str:
        """
        """
        utils.create_dir_if_not_exists(constants.COUNT_VECTORIZER_MODEL_DEFAULT_PATH)
        return constants.COUNT_VECTORIZER_MODEL_DEFAULT_PATH + file_name
    
    @staticmethod
    def _get_default_vocab_path(file_name: str = "/updated_vocab.json") -> str:
        """
        """
        utils.create_dir_if_not_exists(constants.COUNT_VECTORIZER_VOCAB_DEFAULT_PATH)
        return constants.COUNT_VECTORIZER_VOCAB_DEFAULT_PATH + file_name

    @staticmethod
    def _get_loaded_vectorizer_from(path) -> None:
        """
        """
        try:
            loaded_vect = utils.load_data_with_pickle(
                path=path
            )
        except FileNotFoundError:
            logger.warning(
                f"No 'SklearnCountVectorizer' model to load in dir: '{path}'"
            )
            return
        logger.info(
            f"The 'SklearnCountVectorizer' model in '{path}' has been "
            "successfully loaded"
        )
        return loaded_vect

    @staticmethod
    def _get_stored_vocabulary_from(path: str):
        """
        """
        vocab = utils.open_line_by_line_txt_file(
                path=path,
                as_set=True
            )
        if vocab is None:
            vocab = utils.open_json_as_dict(
                path=path
            )
            return vocab
        return vocab

    def _check_if_trained_vectorizer_exists(self) -> bool:
        """
        """
        self.vectorizer = self._get_loaded_vectorizer_from(
            path=self._path_to_trained_model
        )
        if self.vectorizer is None:
            return False
        return True
    
    def _check_if_stored_vocabulary_exists(self) -> bool:
        """
        """
        self.vocab = self._get_stored_vocabulary_from(
            path=self._path_to_stored_vocabulary
        )  
        if self.vocab is None:
            logger.warning(
                f"No 'SklearnCountVectorizer' vocabulary to load in dir: "
                f"'{self._path_to_stored_vocabulary}' or the vocabulary found was "
                "not in the correct format ('.txt' or '.json')"
            )
            return False
        return True
                
    def _set_vocabulary_creator(self) -> None:
        """
        """
        if self._configs.use_own_vocabulary_creator:
            self.vocab = None
            logger.info(
                "'SklearnCountVectorizer' will be ready to create a new "
                "vocabulary from its own parser"
            )
        else:
            # TODO self.vocab = class Vocabulary
            logger.info(
                "A new vocabulary for 'SklearnCountVectorizer' "
                "will be created from an external parser"
            )
   
    def _load_vectorizer_params_from_configs(self) -> None:
        """
        """
        self._max_features = self._configs.max_features
        self._min_ngram = self._configs.min_ngram
        self._max_ngram = self._configs.max_ngram
        
        if not self._check_if_stored_vocabulary_exists():
            self._set_vocabulary_creator()
        
        # if regex_handlers.UppercaseHandler in self.__dict__:
        #     self._lowercase = False
        # else:
        self._lowercase = False
            
        # if self._configs.remove_spanish_stop_words:
        #     self._stop_words = "CONSTANTE_CON_SPANISH_STOP_WORDS"
        # else:
        self._stop_words = None
            
    def _load_vectorizer_params_from_default(self) -> None:
        """
        """
        self._strip_accents = None
        self._analyzer = "word"
        
    def _create_vectorizer(self) -> CountVectorizer:
        """
        """
        return CountVectorizer(
            max_features = self._max_features,
            ngram_range = (self._min_ngram, self._max_ngram),
            vocabulary = self.vocab,
            lowercase = self._lowercase,
            strip_accents = self._strip_accents,
            analyzer = self._analyzer,
            stop_words = self._stop_words
        )

    def _get_vocab_from_vectorizer(self) -> Dict[str, int]:
        """
        """
        return self.vectorizer.vocabulary_

    def _get_vocab_from_trainset(self) -> Set[str]:
        """
        """
        analizer = self.vectorizer.build_analyzer()
        new_vocab = set()
        for sent in self._trainset:
            sent_to_vocab = analizer(sent)
            new_vocab.update(sent_to_vocab)
        return new_vocab

    def _update_loaded_vocabulary(self) -> Dict[str, int]:
        """
        """
        new_vocab = self._get_vocab_from_trainset()
        vocab_to_update = self._get_vocab_from_vectorizer()
        for word in new_vocab:
            if word not in vocab_to_update:
                vocab_to_update[word] = len(vocab_to_update)
        return vocab_to_update

    def _train(self) -> None:
        """
        """
        logger.info("'SklearnCountVectorizer' training has started")
        self.vectorizer.fit(self._trainset)
        logger.info("'SklearnCountVectorizer' training finished")

    def _train_loaded_vectorizer(self) -> None:
        """
        """
        self._train()

    def _train_vectorizer_from_scratch(self) -> None:
        """
        """
        self._load_vectorizer_params_from_configs()
        self._load_vectorizer_params_from_default()
        self.vectorizer = self._create_vectorizer()
        self._train()

    def train(self, trainset: List[str]) -> None:
        """
        """
        self._trainset = trainset

        if self.vectorizer is None:
            if self._check_if_trained_vectorizer_exists():
                self._train_loaded_vectorizer()
            else:
                self._train_vectorizer_from_scratch()
        
        self.persist()

    def load_trained_vectorizer(self, vectorizer: CountVectorizer = None):
        """
        """
        if vectorizer:
            self.vectorizer = vectorizer
            return
        self._check_if_trained_vectorizer_exists()
            
    def persist(self, use_default: bool = False) -> None:
        """
        """
        if use_default:
            path_to_save_vect = self._get_default_model_path()
            logger.info(
                f"The 'SklearnCountVectorizer' trained model will "
                f"be stored in dir: '{path_to_save_vect}'")
        else:
            try:
                path_to_save_vect = self.path_to_save_model + "vocabularies.pkl"
            except TypeError:
                logger.warning(
                    f"No valid path found in '{self.path_to_save_model}' "
                    "to store the 'SklearnCountVectorizer' trained model"
                )
                path_to_save_vect = self._get_default_model_path()
            logger.info(
                f"The 'SklearnCountVectorizer' trained model will be stored " 
                f"in '{path_to_save_vect}'"
                )
                
        utils.persist_data_with_pickle(
            self.vectorizer,
            path_to_save_vect,
            'wb'
        )
        
        if self._update_stored_vocabulary: 
            
            if self.vocab is not None:
                
                updated_vocab = self._update_loaded_vocabulary()

                if use_default:
                    path_to_save_voc = self._get_default_vocab_path()
                    logger.info(
                        f"The updated vocabulary for 'SklearnCountVectorizer' "
                        f"will be stored in dir: '{path_to_save_voc}'"
                    )
                else:
                    try:
                        path_to_save_voc = self.path_to_save_vocabulary + "updated_vocab.json"
                    except TypeError:
                        logger.warning(
                            f"No valid path found in '{self.path_to_save_vocabulary}' "
                            "to store the updated vocabulary for 'SklearnCountVectorizer'"
                        )
                        path_to_save_voc = self._get_default_vocab_path()
                    logger.info(
                        f"The updated vocabulary for 'SklearnCountVectorizer' "
                        f"will be stored in '{path_to_save_voc}'"
                    )
                    
                utils.persist_dict_as_json(
                    updated_vocab,
                    path_to_save_voc
                )
                
            else:
                logger.warning(
                    f"Could not update stored vocabulary for 'SklearnCountVectorizer' "
                    f"because no stored vocabulary was found in '{self._path_to_stored_vocabulary}'"
                )
                return
        
    def process(self, corpus):
        """
        """        
        if self.vectorizer is None:
            logger.warning(
                "It's impossible to process the input from 'SklearnCountVectorizer' "
                "because there is no trained model"
            )
            return corpus

        corpus = self.vectorizer.transform(corpus)
        
        return corpus
            

