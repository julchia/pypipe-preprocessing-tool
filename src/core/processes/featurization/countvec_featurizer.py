from __future__ import annotations
from typing import List, Set, Dict

import logging
from omegaconf import OmegaConf
from sklearn.feature_extraction.text import CountVectorizer

from src.core.processes import utils
from src.core.processes.featurization.featurizers import TextFeaturizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CountVecFeaturizer(TextFeaturizer):
    """
    """
    
    def __init__(
        self,
        configs: OmegaConf, 
        featurizer: CountVectorizer = None,
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
        
        self._path_to_get_trained_model = self._configs.path_to_get_trained_model
        self._path_to_get_stored_vocabulary = self._configs.path_to_get_stored_vocabulary
        
        self._update_stored_vocabulary = self._configs.update_stored_vocabulary

    @classmethod
    def get_isolated_process(
        cls, 
        configs: OmegaConf, 
        featurizer: CountVectorizer = None,
        alias: str = None,
    ) -> CountVecFeaturizer:
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
            "max_features": None,
            "min_ngram": 1,
            "max_ngram": 1,
            "remove_spanish_stop_words": False,
            "path_to_get_trained_model": None,
            "path_to_get_stored_vocabulary": None,
            "update_stored_vocabulary": False,
            "use_own_vocabulary_creator": True,
            "unk_token": "<<UNK>>",
            "path_to_save_model": None,
            "path_to_save_vocabulary": None
        })

    def _check_if_trained_featurizer_exists_and_load_it(self) -> bool:
        """
        """
        self.featurizer = TextFeaturizer.data_manager.load_data_from_callable(
            "rb",
            callback_fn_to_load_data=utils.load_data_with_pickle,
            path_to_load_data=self._path_to_get_trained_model
        )
        if self.featurizer is None:
            return False
        return True
    
    def _check_if_stored_vocabulary_exists_and_load_it(self) -> bool:
        """
        """
        path = self._path_to_get_stored_vocabulary
        if TextFeaturizer.data_manager.check_if_dir_extension_is('.json', path):
            self.vocab = TextFeaturizer.data_manager.load_data_from_callable(
                callback_fn_to_load_data=utils.open_json_as_dict,
                path_to_load_data=path
            )
            return True
        elif TextFeaturizer.data_manager.check_if_dir_extension_is('.txt', path):
            self.vocab = TextFeaturizer.data_manager.load_data_from_callable(
                callback_fn_to_load_data=utils.open_line_by_line_txt_file,
                path_to_load_data=path
            )
            return True
        else:
            logger.warning(
                f"No 'CountVecFeaturizer' vocabulary to load in dir: "
                f"'{path}', or the vocabulary file was not in the correct "
                f"format: '.txt' or '.json'"
            )
            return False
                
    def _set_vocabulary_creator(self) -> None:
        """
        """
        if self._configs.use_own_vocabulary_creator:
            if self._unk_token is not None:
                self._trainset.append(self._unk_token)
            self.vocab = None
            logger.info(
                "'CountVecFeaturizer' will be ready to create a new "
                "vocabulary from its own parser"
            )
        else:
            self.vocab = super().create_vocab(
                corpus=self._trainset,
                unk_text=self._unk_token
            )
            logger.info(
                "A new vocabulary for 'CountVecFeaturizer' "
                "will be created from Vocabulary object"
            )
   
    def _load_featurizer_params_from_configs(self) -> None:
        """
        """
        if not self._check_if_stored_vocabulary_exists_and_load_it():
            self._set_vocabulary_creator()        

        self._max_features = self._configs.max_features
        self._min_ngram = self._configs.min_ngram
        self._max_ngram = self._configs.max_ngram
        self._lowercase = False
        self._stop_words = None
            
    def _load_featurizer_params_from_default(self) -> None:
        """
        """
        self._strip_accents = None
        self._analyzer = "word"
        
    def _create_featurizer(self) -> CountVectorizer:
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

    def _get_vocab_from_featurizer(self) -> Dict[str, int]:
        """
        """
        return self.featurizer.vocabulary_

    def _get_vocab_from_trainset(self) -> Set[str]:
        """
        """
        analizer = self.featurizer.build_analyzer()
        new_vocab = set()
        for sent in self._trainset:
            sent_to_vocab = analizer(sent)
            new_vocab.update(sent_to_vocab)
        return new_vocab

    def _update_loaded_vocabulary(self) -> Dict[str, int]:
        """
        """
        new_vocab = self._get_vocab_from_trainset()
        vocab_to_update = self._get_vocab_from_featurizer()
        for word in new_vocab:
            if word not in vocab_to_update:
                vocab_to_update[word] = len(vocab_to_update)
        return vocab_to_update

    def _train(self) -> None:
        """
        """
        logger.info("'CountVecFeaturizer' training has started")
        self.featurizer.fit(self._trainset)
        logger.info("'CountVecFeaturizer' training finished")

    def _train_loaded_featurizer(self) -> None:
        """
        """
        self._train()

    def _train_featurizer_from_scratch(self) -> None:
        """
        """
        self._load_featurizer_params_from_configs()
        self._load_featurizer_params_from_default()
        self.featurizer = self._create_featurizer()
        self._train()

    def train(self, trainset: List[str], persist: bool = False) -> None:
        """
        """
        self._trainset = trainset

        if self.featurizer is None:
            if self._check_if_trained_featurizer_exists_and_load_it():
                self._train_loaded_featurizer()
            else:
                self._train_featurizer_from_scratch()
        else:
            self._train_loaded_featurizer()
            
        if persist:
            self.persist()

    def load(self, featurizer: CountVectorizer = None):
        """
        """
        if featurizer is not None:
            self.featurizer = featurizer
        else:
            self._check_if_trained_featurizer_exists_and_load_it()
            
    def persist(self) -> None:
        """
        """
        # try to save model
        TextFeaturizer.data_manager.save_data_from_callable(
            self.featurizer,
            "wb",
            callback_fn_to_save_data=utils.persist_data_with_pickle,
            path_to_save_data=self.path_to_save_model,
            data_file_name="/vocabularies.pkl",
            alias="countvec_featurizer"
        )
            
        # try to save vocab data
        if self._update_stored_vocabulary: 
            if self.vocab is not None:
                updated_vocab = self._update_loaded_vocabulary()
                TextFeaturizer.data_manager.save_data_from_callable(
                    updated_vocab,
                    "w",
                    callback_fn_to_save_data=utils.persist_dict_as_json,
                    path_to_save_data=self.path_to_save_vocabulary,
                    data_file_name="/vocab.json",
                    alias="countvec_featurizer",
                    to_save_vocab=True
                )
            else:
                logger.warning(
                    f"Could not update stored vocabulary for 'CountVecFeaturizer' "
                    f"'CountVecFeaturizer' because no stored vocabulary was found "
                    f"in '{self._path_to_get_stored_vocabulary}'"
                )
                return
        else:
            vec_vocab = self._get_vocab_from_featurizer()  
            TextFeaturizer.data_manager.save_data_from_callable(
                vec_vocab,
                "w",
                callback_fn_to_save_data=utils.persist_dict_as_json,
                path_to_save_data=self.path_to_save_vocabulary,
                data_file_name="/vocab.json",
                alias="countvec_featurizer",
                to_save_vocab=True
            )
                
    def process(self, corpus):
        """
        """        
        if self.featurizer is None:
            logger.warning(
                "It's impossible to process the input from 'CountVecFeaturizer' "
                "because there is no trained model"
            )
            return corpus

        corpus = self.featurizer.transform(corpus)
        
        return corpus

