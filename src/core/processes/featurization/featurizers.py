from __future__ import annotations
from typing import List, Dict, Set, Union, Optional

import logging
from omegaconf import OmegaConf, DictConfig
from numpy import ndarray
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec, KeyedVectors

from src.core import constants
from src.core.processes import utils
from src.core.processes.featurization.base import TextFeaturizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CountVecFeaturizer(TextFeaturizer):
    """
    Sparse feautrizer based on Sklearn CountVectorizer.
    
    Converts a collection of text documents to a matrix of token counts.
    The implementation produces a sparse representation of the counts 
    using scipy.sparse.csr_matrix.
    """
    def __init__(
        self,
        configs: OmegaConf, 
        featurizer: CountVectorizer = None,
        alias: str = None
    ) -> None:
        """
        Builds a CountVecFeaturizer object by taking configurations
        from the configs object.
        
        Args:
            config: Featurizer configurations.
        
            featurizer: Trained or non-trained CountVectorizer object.
                If the featurizer is not trained, it will be trained 
                with the 'config' settings. If it is trained, when you 
                re-train it will do so with the 'config' settings.
        
            alias: Alias to recognize the featurizer within a 
                pipeline (it is None if the featurizer is not 
                within a pipeline).
        """
        super().__init__(
            configs=configs,
            alias=alias
        )
        
        self._alias = constants.COUNTVEC_FEATURIZER_ALIAS if alias is None else alias
        
        self.featurizer = featurizer
        
        self._unk_token = self._configs.unk_token
        
        self.path_to_save_model = self._configs.path_to_save_model 
        self.path_to_save_vocabulary = self._configs.path_to_save_vocabulary
        
        self._path_to_get_trained_model = self._configs.path_to_get_trained_model
        self._path_to_get_stored_vocabulary = self._configs.path_to_get_stored_vocabulary
        
        self._use_own_vocabulary_creator = self._configs.use_own_vocabulary_creator
        
        self._update_stored_vocabulary = self._configs.update_stored_vocabulary

    @classmethod
    def get_isolated_process(
        cls, 
        configs: OmegaConf, 
        featurizer: CountVectorizer = None,
        alias: str = None,
    ) -> CountVecFeaturizer:
        """Returns non-trained CountVecFeaturizer object"""
        return cls(
            configs=configs,
            featurizer=featurizer,
            alias=alias
        )
        
    @classmethod
    def get_default_configs(cls) -> DictConfig:
        """Returns configurations for CountVecFeaturizer object"""
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
        """Checks if a trained featurizer was set in the configurations 
        and loads it if present."""
        self.featurizer = TextFeaturizer.data_manager.load_data_from_callable(
            "rb",
            callback_fn_to_load_data=utils.load_data_with_pickle,
            path_to_load_data=self._path_to_get_trained_model
        )
        return self.featurizer is not None
    
    def _check_if_stored_vocabulary_exists_and_load_it(self) -> bool:
        """Checks if a vocabulary was set in the configurations and
        loads it if present."""
        path = self._path_to_get_stored_vocabulary
        
        if utils.check_if_dir_extension_is('.json', path):
            self.vocab = TextFeaturizer.data_manager.load_data_from_callable(
                callback_fn_to_load_data=utils.open_json_as_dict,
                path_to_load_data=path
            )
        elif utils.check_if_dir_extension_is('.txt', path):
            self.vocab = TextFeaturizer.data_manager.load_data_from_callable(
                callback_fn_to_load_data=utils.open_line_by_line_txt_file,
                path_to_load_data=path
            )
        else:
            logger.warning(
                f"No 'CountVecFeaturizer' vocabulary to load in dir: "
                f"'{path}', or the vocabulary file was not in the correct "
                f"format: '.txt' or '.json'"
            )
            return False
        
        return self.vocab is not None
                
    def _set_vocabulary_creator(self) -> None:
        """
        Sets the way to create the vocabulary from the trainset corpus.
        The vocabulary can be created either from the CountVectorizer by 
        setting 'self.vocab' to None, or from the Vocabulary object by 
        creating a generator.
        """
        if self._use_own_vocabulary_creator:
            # if self._unk_token is not None:
            #     self._trainset.append(self._unk_token)
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
   
    def _load_featurizer_params(self) -> None:
        """Loads necessary parameters to create CountVectorizer object."""
        if not self._check_if_stored_vocabulary_exists_and_load_it():
            self._set_vocabulary_creator()
        
        self._featurizer_params = {
            "max_features": self._configs.max_features,
            "ngram_range": (self._configs.min_ngram, self._configs.max_ngram),
            "vocabulary": self.vocab,
            "lowercase": False,
            "stop_words": None,
            "strip_accents": None,
            "analyzer": "word"
        }
                    
    def _create_featurizer(self) -> CountVectorizer:
        """Returns non-trained CountVectorizer object."""
        return CountVectorizer(**self._featurizer_params)

    def _get_vocab_from_featurizer(self) -> Dict[str, int]:
        """Returns the vocabulary that generates CountVectorizer from 
        loaded vocabulary."""
        return self.featurizer.vocabulary_

    def _get_vocab_from_trainset(self) -> Set[str]:
        """Returns the vocabulary that generates CountVectorizer from 
        trainset."""
        analizer = self.featurizer.build_analyzer()
        new_vocab = set()
        for sent in self._trainset:
            sent_to_vocab = analizer(sent)
            new_vocab.update(sent_to_vocab)
        return new_vocab

    def _update_loaded_vocabulary(self) -> Dict[str, int]:
        """Updates the loaded vocabulary by adding any non-seen words from 
        trainset."""
        new_vocab = self._get_vocab_from_trainset()
        vocab_to_update = self._get_vocab_from_featurizer()
        for word in new_vocab:
            if word not in vocab_to_update:
                vocab_to_update[word] = len(vocab_to_update)
        return vocab_to_update

    def _train(self) -> None:
        """Fits CountVectorizer object."""
        logger.info("'CountVecFeaturizer' training has started")        
        self.featurizer.fit(self._trainset)
        logger.info("'CountVecFeaturizer' training finished")

    def _train_loaded_featurizer(self) -> None:
        self._train()

    def _train_featurizer_from_scratch(self) -> None:
        self._load_featurizer_params()
        self.featurizer = self._create_featurizer()
        self._train()

    def train(self, trainset: List[str], persist: bool = False) -> None:
        """
        Interface that trains CountFeaturizer object.
        
        Before training, check if there is a loaded featurizer. If it 
        does not exist, train from scratch.
        
        args:
            trainset: training corpus, a list of sentences.
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
            self.persist(model=True, vocab=True)

    def load(self, featurizer: CountVectorizer = None):
        """
        Loads CountVectorizer object.
        
        args:
            featurizer: Trained or non-trained CountVectorizer object.
                If the featurizer is not trained, it will be trained 
                with the 'config' settings. If it is trained, when you 
                re-train it will do so with the 'config' settings.
        """
        if isinstance(featurizer, CountVectorizer):
            self.featurizer = featurizer
        else:
            self._check_if_trained_featurizer_exists_and_load_it()
    
    def _persist_vocab(self, vocab: Dict[int, str]) -> None:
        TextFeaturizer.data_manager.save_data_from_callable(
            vocab,
            "w",
            callback_fn_to_save_data=utils.persist_dict_as_json,
            path_to_save_data=self.path_to_save_vocabulary,
            data_file_name="/vocab.json",
            alias=self._alias,
            to_save_vocab=True
        )
        
    def _persist_model(self, model: CountVectorizer) -> None:
        TextFeaturizer.data_manager.save_data_from_callable(
            model,
            "wb",
            callback_fn_to_save_data=utils.persist_data_with_pickle,
            path_to_save_data=self.path_to_save_model,
            data_file_name="/vocabularies.pkl",
            alias=self._alias
        )
            
    def persist(self, model=True, vocab=False) -> None:
        if model:
            self._persist_model(model=self.featurizer)
            
        if vocab:
            if self._update_stored_vocabulary: 
                if self.vocab is not None:
                    updated_vocab = self._update_loaded_vocabulary()
                    self._persist_vocab(vocab=updated_vocab)
                else:
                    logger.warning(
                        f"Could not update stored vocabulary for 'CountVecFeaturizer' "
                        f"'CountVecFeaturizer' because no stored vocabulary was found "
                        f"in '{self._path_to_get_stored_vocabulary}'"
                    )
                    return
            else:
                vec_vocab = self._get_vocab_from_featurizer()
                self._persist_vocab(vocab=vec_vocab)
    
    def process(self, corpus: List[str]) -> Union(List[str], ndarray):
        """
        Process a corpus of text and, if there is a trained CountVectorizer 
        object, converts it to a matrix of token counts.

        Args:
            corpus: list of sentences. If None, returns the original corpus 
            without processing.
        """
        if self.featurizer is None:
            logger.warning(
                "It's impossible to process the input from 'CountVecFeaturizer' "
                "because there is no trained model"
            )
            return corpus

        corpus = self.featurizer.transform(corpus)
        
        return corpus
    

class Word2VecFeaturizer(TextFeaturizer):
    """
    Dense feautrizer based on Gensim 4.x Word2Vec.
    
    The Gensim word2vec algorithms include skip-gram and CBOW models, using
    either hierarchical softmax or negative sampling.
    """
    def __init__(
        self,
        configs: OmegaConf, 
        featurizer: Word2Vec = None,
        alias: str = None
    ) -> None:
        """
        Builds a Word2VecFeaturizer object by taking configurations
        from the configs object.
        
        Args:
            config: Featurizer configurations.
        
            featurizer: Trained or non-trained Word2Vec object.
                If the featurizer is not trained, it will be trained 
                with the 'config' settings. If it is trained, when you 
                re-train it will do so with the 'config' settings.
        
            alias: Alias to recognize the featurizer within a 
                pipeline (it is None if the featurizer is not 
                within a pipeline).
        """
        super().__init__(
            configs=configs,
            alias=alias
        )
        
        self._alias = constants.WORD2VEC_FEATURIZER_ALIAS if alias is None else alias
        
        self.featurizer = featurizer
        
        self._unk_token = self._configs.unk_token
        
        self._update_stored_vocabulary = self._configs.update_stored_vocabulary
        
        self.path_to_save_model = self._configs.path_to_save_model
        self.path_to_save_vocabulary = self._configs.path_to_save_vocabulary
        self.path_to_save_vectors = self._configs.path_to_save_vectors
        
        self._path_to_trained_model = self._configs.path_to_get_trained_model
        self._path_to_get_trained_vectors = self._configs.path_to_get_trained_vectors
        self._path_to_get_stored_vocabulary = self._configs.path_to_get_stored_vocabulary
                
        if self._path_to_get_stored_vocabulary is not None:
            if self._path_to_get_trained_vectors is not None:
                logger.warning(
                    "In 'Word2VecFeaturizer' config file, one path has "
                    "detected for load vocabulary and another for load "
                    "pre-trained vectors. For model training, the pre-trained "
                    "vectors will be prioritized. If you want to use "
                    "the vocabulary, remove the path for pre-trained "
                    "vectors from the config file."
                )
                self._path_to_get_stored_vocabulary = None
        
    @classmethod
    def get_isolated_process(
        cls,
        configs: OmegaConf, 
        featurizer: Word2Vec = None,
        alias: str = None
    ) -> Word2VecFeaturizer:
        """Returns non-trained Word2VecFeaturizer object"""
        return cls(
            configs=configs,
            featurizer=featurizer,
            alias=alias
        )
        
    @classmethod
    def get_default_configs(cls) -> OmegaConf:
        """Returns configurations for Word2VecFeaturizer object"""
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
        """Checks if a trained featurizer was set in the configurations 
        and loads it if present."""
        self.featurizer = TextFeaturizer.data_manager.load_data_from_callable(
            callback_fn_to_load_data=Word2Vec.load,
            path_to_load_data=self._path_to_trained_model
        )
        return self.featurizer is not None
        
    def _load_featurizer_params(self) -> None:        
        """Loads necessary parameters to create Word2Vec object."""
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
        """Loads necessary parameters to train Word2Vec object."""
        self._train_params = {
            "corpus_iterable": self._vocab,
            "total_examples": self.featurizer.corpus_count,
            "epochs": self._configs.epochs,
            "start_alpha": None,
            "end_alpha": None
        }
    
    def _load_vectors(self, path_to_vectors: str) -> KeyedVectors:
        """Loads KeyedVectors object from configuratios with a representation 
        of trained dense vectors."""
        if utils.check_if_dir_extension_is('.bin', path_to_vectors):
            return KeyedVectors.load_word2vec_format(path_to_vectors, binary=True)
        else:
            return KeyedVectors.load(path_to_vectors)
    
    def _prepare_vocabulary_from_loaded_vocab(self, update: bool) -> None:
        """Loads word-freq dictionary from configuratios and and sets it 
        as Word2Vec vocab."""
        from json import load
        
        with open(self._path_to_get_stored_vocabulary, 'r') as f:
            stored_vocab = load(f)
            
        self.featurizer.build_vocab_from_freq(
            word_freq=stored_vocab, 
            update=update
        )  
    
    def _prepare_vocabulary_from_pretrained_vectors(self, update: bool) -> None:
        """Loads pretrained vectors from configuratios and sets it as Word2Vec
        vocab."""
        wv = self._load_vectors(self._path_to_get_trained_vectors)
        
        self.featurizer.build_vocab_from_freq(
            word_freq=wv.key_to_index, 
            update=update
        )
        
        self.featurizer.wv.vectors = wv.vectors
        
    def _prepare_vocabulary(self) -> None:
        """
        Prepares the vocabulary to initialize the Word2Vec object. The vocabulary 
        can be loaded in different ways: 
        - from pre-trained dense vectors
        - from a frequency dictionary
        - from the generator created by the Vocabulary object.
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
               
        if self._path_to_get_trained_vectors is not None:
            self._prepare_vocabulary_from_pretrained_vectors(update)
        elif self._path_to_get_stored_vocabulary is not None:
            self._prepare_vocabulary_from_loaded_vocab(update)
        else:       
            self.featurizer.build_vocab(
                corpus_iterable=self._vocab,
                update=update
            )
    
    def _create_featurizer(self) -> Word2Vec:
        """Returns non-trained Word2Vec object."""
        return Word2Vec(**self._featurizer_params)
    
    def _train(self) -> None:
        """Trains Word2Vec object."""
        self._prepare_vocabulary()
        self._load_train_params()
        logger.info("'Word2VecFeaturizer' training has started")
        self.featurizer.train(**self._train_params)
        logger.info("'Word2VecFeaturizer' training finished")
    
    def _train_loaded_featurizer(self) -> None:
        self._train()
        
    def _train_featurizer_from_scratch(self) -> None:
        self._load_featurizer_params()
        self.featurizer = self._create_featurizer()
        self._train()

    def train(self, trainset: List[str], persist: bool = False) -> None:
        """
        Interface that trains Word2Vec object.
        
        Before training, check if there is a loaded featurizer. If it 
        does not exist, train from scratch.
        
        args:
            trainset: training corpus, a list of sentences.
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
            self.persist(model=True, vocab=True, vectors=True)

    def load(self, path_to_model: str = None) -> None:
        """
        Loads Word2Vec object.
        
        args: 
            path_to_model: Path to trained or non-trained Word2Vec 
                object. If the featurizer is not trained, it will 
                be trained with the 'config' settings. If it is 
                trained, when you re-train it will do so with the
                'config' settings.
        """
        if path_to_model is not None:
            self.featurizer = Word2Vec.load(
                fname=path_to_model
            )
        else:
            self._check_if_trained_featurizer_exists_and_load_it()
    
    def persist(
        self, 
        model: bool = True, 
        vocab: bool = False, 
        vectors: bool = False
    ) -> None:
        if model:
            TextFeaturizer.data_manager.save_data_from_callable(
                callback_fn_to_save_data=self.featurizer.save,
                path_to_save_data=self.path_to_save_model,
                data_file_name="/word2vec_model.model",
                alias=self._alias
            )
        
        if vocab:
            TextFeaturizer.data_manager.save_data_from_callable(
                self.featurizer.wv.key_to_index,
                "w",
                callback_fn_to_save_data=utils.persist_dict_as_json,
                path_to_save_data=self.path_to_save_vocabulary,
                data_file_name="/word2vec_vocab.json",
                alias=self._alias,
                to_save_vocab=True
            )
            
        if vectors:
            TextFeaturizer.data_manager.save_data_from_callable(
                callback_fn_to_save_data=self.featurizer.wv.save,
                path_to_save_data=self.path_to_save_vectors,
                data_file_name="/word2vec_vectors.kv",
                alias=self._alias,
                to_save_vocab=True
            )
                
    def load_vectors(self, path_to_vectors: str) -> KeyedVectors:
        """
        Interface to load KeyedVectors object from configuratios with a 
        representation of trained dense vectors.
        
        args:
            path_to_vectors: path to get vectors.
        """
        return self._load_vectors(path_to_vectors)
                                       
    def get_word_vector_object(self) -> Optional[KeyedVectors]:
        """Returns KeyedVectors object (see Gensim 4.x models.keyedvectors).
        If there is no trained featurizer, returns None."""
        if self.featurizer is None:
            logger.warning(
                "It's impossible to get 'KeyedVectors' from 'Word2VecFeaturizer' "
                "object because there is no trained model"
            )
            return
        return self.featurizer.wv
    
    def get_vector_by_key(
        self, 
        key: Union[str, List[str]]
    ) -> Union(str, List[str], ndarray):
        """
        Given a word learned during training, returns its corresponding dense vector
        (1D if key_or_keys is single key, otherwise - 2D).

        If there is no trained featurizer, returns None.

        Args:
            key: requested key or list-of-keys.
        """
        if self.featurizer is None:
            logger.warning(
                "It's impossible to get vector items from 'Word2VecFeaturizer' "
                "object because there is no trained model"
            )
            return
        else:
            return self.featurizer.wv.__getitem__(key)
        
