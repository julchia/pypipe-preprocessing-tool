import os
from typing import List, Dict

import pytest
from pytest_mock import mocker

from omegaconf import OmegaConf, DictConfig
from sklearn.feature_extraction.text import CountVectorizer

from pypipe.tests import utils
from pypipe.core.management.managers import VocabularyManager
from pypipe.core.processes.featurization.base import TextFeaturizer
from pypipe.core.processes.featurization.featurizers import CountVecFeaturizer


######################################################################################
###################################### Fixtures ######################################
######################################################################################


@pytest.fixture
def get_dummy_dict_vocab_for_testing() -> Dict[int, str]:
    return {0: "file", 1: "for", 2: "testing"}
    

@pytest.fixture
def get_dummy_list_vocab_for_testing() -> List[str]:
    return ["file", "for", "testing"]


@pytest.fixture
def get_mocked_logger_at_info_level(mocker):
    return mocker.patch("logging.Logger.info")


@pytest.fixture
def get_mocked_logger_at_warning_level(mocker):
    return mocker.patch("logging.Logger.warning")


@pytest.fixture
def get_OmegaConf_instance_for_test_CountVecFeaturizer() -> DictConfig:
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


@pytest.fixture
def get_CountVecFeaturizer_instance_for_testing(
    get_OmegaConf_instance_for_test_CountVecFeaturizer
) -> CountVecFeaturizer:
    config = get_OmegaConf_instance_for_test_CountVecFeaturizer
    return CountVecFeaturizer.get_isolated_process(configs=config)


@pytest.fixture
def get_CountVectorizer_instance_for_testing(
    get_CountVecFeaturizer_instance_for_testing,
    get_dummy_dict_vocab_for_testing
) -> CountVectorizer:
    featurizer = get_CountVecFeaturizer_instance_for_testing
    
    featurizer._featurizer_params = {
        "max_features": 100,
        "ngram_range": (1, 2),
        "vocabulary": get_dummy_dict_vocab_for_testing,
        "lowercase": False,
        "stop_words": None,
        "strip_accents": None,
        "analyzer": "word"
    }
    
    return featurizer._create_featurizer()


###################################################################################
###################################### Tests ###################################### 
###################################################################################


################## CountVecFeaturizer ##################
    

def test_get_isolated_process_class_method_when_config_is_given_and_vectorizer_is_None_expected_CountVecFeaturizer_obj_type_with_correct_attrs(
    get_OmegaConf_instance_for_test_CountVecFeaturizer,
):
    config = get_OmegaConf_instance_for_test_CountVecFeaturizer
    
    isolated_process = CountVecFeaturizer.get_isolated_process(
        configs=config, 
        featurizer=None,
        alias=None
    )
    
    assert isinstance(isolated_process, CountVecFeaturizer)
    assert "configs" not in isolated_process.__dict__
    assert "alias" not in isolated_process.__dict__
    assert isolated_process.featurizer == None
    

def test_get_isolated_process_class_method_when_config_is_given_and_vectorizer_is_not_None_expected_CountVecFeaturizer_obj_type_with_correct_attrs(
    get_OmegaConf_instance_for_test_CountVecFeaturizer,
):
    config = get_OmegaConf_instance_for_test_CountVecFeaturizer
    featurizer = CountVectorizer()
    
    isolated_process = CountVecFeaturizer.get_isolated_process(
        configs=config, 
        featurizer=featurizer,
        alias=None
    )
    
    assert isinstance(isolated_process, CountVecFeaturizer)
    assert "configs" not in isolated_process.__dict__
    assert "alias" not in isolated_process.__dict__
    assert isinstance(isolated_process.featurizer, CountVectorizer)
    

def test_get_default_configs_class_method_expected_DictConfig_obj_type(
):
    config = CountVecFeaturizer.get_default_configs()
            
    assert isinstance(config, DictConfig)
    

def test__check_if_trained_featurizer_exists_and_load_it_method_when__path_to_get_trained_model_is_pkl_file_expected_True(
    get_CountVecFeaturizer_instance_for_testing
):        
    temp_pkl_file = utils.create_temp_pickle_file(
        obj=CountVectorizer(),
        suffix='.pkl',
        delete=False
    )
    
    featurizer = get_CountVecFeaturizer_instance_for_testing
    featurizer._path_to_get_trained_model = temp_pkl_file
    
    assert featurizer._check_if_trained_featurizer_exists_and_load_it() == True
    
    os.remove(temp_pkl_file)
    
    
def test__check_if_trained_featurizer_exists_and_load_it_method_when__path_to_get_trained_model_is_None_expected_False(
    get_CountVecFeaturizer_instance_for_testing
):        
    featurizer = get_CountVecFeaturizer_instance_for_testing
    featurizer._path_to_get_trained_model = None
    
    assert featurizer._check_if_trained_featurizer_exists_and_load_it() == False
    

def test__check_if_stored_vocabulary_exists_and_load_it_method_when__path_to_get_stored_vocabulary_is_json_file_expected_True(
    get_dummy_dict_vocab_for_testing,
    get_CountVecFeaturizer_instance_for_testing
):                 
    tmp_json_file = utils.create_temp_json_file_from_dict(
        obj=get_dummy_dict_vocab_for_testing, 
        delete=False
    )
            
    featurizer = get_CountVecFeaturizer_instance_for_testing
    featurizer._path_to_get_stored_vocabulary = tmp_json_file
    
    assert featurizer._check_if_stored_vocabulary_exists_and_load_it() == True
    
    os.remove(tmp_json_file)
    

def test__check_if_stored_vocabulary_exists_and_load_it_method_when__path_to_get_stored_vocabulary_is_txt_file_expected_True(
    get_dummy_list_vocab_for_testing,
    get_CountVecFeaturizer_instance_for_testing
):
    tmp_txt_file = utils.create_temp_txt_file_from_list(
        obj=get_dummy_list_vocab_for_testing, 
        delete=False
    )
            
    featurizer = get_CountVecFeaturizer_instance_for_testing
    featurizer._path_to_get_stored_vocabulary = tmp_txt_file
    
    assert featurizer._check_if_stored_vocabulary_exists_and_load_it() == True
    
    os.remove(tmp_txt_file)
    

def test__check_if_stored_vocabulary_exists_and_load_it_method_when__path_to_get_stored_vocabulary_is_not_json_or_txt_file_expected_False(
    get_CountVecFeaturizer_instance_for_testing
):
    temp_bin_file = utils.create_temp_pickle_file(
        obj=CountVectorizer(),
        suffix='.bin',
        delete=False
    )
    
    featurizer = get_CountVecFeaturizer_instance_for_testing
    featurizer._path_to_get_stored_vocabulary = temp_bin_file
    
    assert featurizer._check_if_stored_vocabulary_exists_and_load_it() == False
    
    os.remove(temp_bin_file)


def test__check_if_stored_vocabulary_exists_and_load_it_method_when__path_to_get_stored_vocabulary_is_nonexistent_file_expected_False(
    get_CountVecFeaturizer_instance_for_testing
):
    featurizer = get_CountVecFeaturizer_instance_for_testing
    featurizer._path_to_get_stored_vocabulary = "nonexistent_file.json"
    
    assert featurizer._check_if_stored_vocabulary_exists_and_load_it() == False


def test_check_if_stored_vocabulary_exists_and_load_it_method_when__path_to_get_stored_vocabulary_is_None_expected_False(
    get_CountVecFeaturizer_instance_for_testing
):
    featurizer = get_CountVecFeaturizer_instance_for_testing
    featurizer._path_to_get_stored_vocabulary = None
    
    assert featurizer._check_if_stored_vocabulary_exists_and_load_it() == False


def test__set_vocabulary_creator_method_when_use_own_vocabulary_creator_is_True_expected_vocab_set_to_None(
    get_CountVecFeaturizer_instance_for_testing
):
    featurizer = get_CountVecFeaturizer_instance_for_testing
    featurizer._configs.use_own_vocabulary_creator = True
    
    featurizer._set_vocabulary_creator()
    
    assert featurizer.vocab == None


def test__set_vocabulary_creator_when_use_own_vocabulary_creator_is_False_and__trainset_is_given_expected_Vocabulary_obj_type(
    get_CountVecFeaturizer_instance_for_testing,
    get_dummy_list_vocab_for_testing
):        
    featurizer = get_CountVecFeaturizer_instance_for_testing
    featurizer._use_own_vocabulary_creator = False
    featurizer._trainset = get_dummy_list_vocab_for_testing
            
    featurizer._set_vocabulary_creator()
    
    assert isinstance(featurizer.vocab, VocabularyManager)


def test__set_vocabulary_creator_method_when_use_own_vocabulary_creator_is_False_and__trainset_is_given_expected_supercall_create_vocab(
    get_CountVecFeaturizer_instance_for_testing,
    get_dummy_list_vocab_for_testing,
    mocker
):        
    countvec = get_CountVecFeaturizer_instance_for_testing
    countvec._use_own_vocabulary_creator = False
    countvec._trainset = get_dummy_list_vocab_for_testing
    countvec._unk_token = None
    
    mocked_create_vocab_super_method = mocker.patch.object(
        TextFeaturizer, 
        "create_vocab"
    )
    
    countvec._set_vocabulary_creator()
    
    mocked_create_vocab_super_method.assert_called_once_with(
        corpus=countvec._trainset, 
        unk_text=countvec._unk_token
    )


def test__load_featurizer_params_method_expected_correct_featurizer_params(
    get_CountVecFeaturizer_instance_for_testing,
    mocker
):
    featurizer = get_CountVecFeaturizer_instance_for_testing
    
    mocker.patch.object(featurizer, "_check_if_stored_vocabulary_exists_and_load_it")
    mocker.patch.object(featurizer, "_set_vocabulary_creator")
    
    featurizer._configs.max_features = 100
    featurizer._configs.min_ngram = 1
    featurizer._configs.max_ngram = 2
    featurizer.vocab = None
    featurizer._featurizer_params = None
    
    featurizer._load_featurizer_params()
    
    expected_params = {
        "max_features": 100,
        "ngram_range": (1, 2),
        "vocabulary": None,
        "lowercase": False,
        "stop_words": None,
        "strip_accents": None,
        "analyzer": "word"
    }
    
    assert featurizer._featurizer_params == expected_params


def test__load_featurizer_params_method_when_vocab_notNone_so__check_if_stored_vocabulary_exists_and_load_it_method_is_called_with_True_expected_correct_calls(
    get_CountVecFeaturizer_instance_for_testing,
    get_dummy_dict_vocab_for_testing,
    mocker
):
    featurizer = get_CountVecFeaturizer_instance_for_testing
    
    featurizer.vocab = get_dummy_dict_vocab_for_testing
    
    mocked_check_if_stored_vocabulary_exists_and_load_it_method = mocker.patch.object(
        featurizer, "_check_if_stored_vocabulary_exists_and_load_it",
        return_value=True
    )
    
    mocked_set_vocabulary_creator_method = mocker.patch.object(
        featurizer, "_set_vocabulary_creator"
    )
    
    featurizer._load_featurizer_params()
    
    mocked_check_if_stored_vocabulary_exists_and_load_it_method.assert_called_once()
    mocked_set_vocabulary_creator_method.assert_not_called()


def test__load_featurizer_params_method_when_vocab_None_so__check_if_stored_vocabulary_exists_and_load_it_method_is_called_with_False_expected_correct_calls(
    get_CountVecFeaturizer_instance_for_testing,
    mocker
):
    featurizer = get_CountVecFeaturizer_instance_for_testing
    
    featurizer.vocab = None
    
    mocked_check_if_stored_vocabulary_exists_and_load_it_method = mocker.patch.object(
        featurizer, "_check_if_stored_vocabulary_exists_and_load_it",
        return_value=False
    )
    
    mocked_set_vocabulary_creator_method = mocker.patch.object(
        featurizer, "_set_vocabulary_creator"
    )
    
    featurizer._load_featurizer_params()
    
    mocked_check_if_stored_vocabulary_exists_and_load_it_method.assert_called_once()    
    mocked_set_vocabulary_creator_method.assert_called_once()
    

def test__create_featurizer_expected_CountVectorizer_obj_type(
    get_CountVecFeaturizer_instance_for_testing
):
    featurizer = get_CountVecFeaturizer_instance_for_testing
    
    featurizer._featurizer_params = {
        "max_features": 100,
        "ngram_range": (1, 2),
        "vocabulary": None,
        "lowercase": False,
        "stop_words": None,
        "strip_accents": None,
        "analyzer": "word"
    }
    
    model = featurizer._create_featurizer()
    
    assert isinstance(model, CountVectorizer)


def test__get_vocab_from_featurizer_method_expected_dict_obj_type(
    get_CountVecFeaturizer_instance_for_testing,
    get_CountVectorizer_instance_for_testing,
    get_dummy_dict_vocab_for_testing
):
    vocab = get_dummy_dict_vocab_for_testing

    countvec = get_CountVecFeaturizer_instance_for_testing
    countvec.featurizer = get_CountVectorizer_instance_for_testing
    countvec.featurizer.vocabulary_ = vocab

    returned_vocab = countvec._get_vocab_from_featurizer()

    assert isinstance(returned_vocab, dict)
    assert returned_vocab == vocab


def test__get_vocab_from_trainset_method_expected_set_obj_type(
    get_CountVecFeaturizer_instance_for_testing,
    get_CountVectorizer_instance_for_testing,
    get_dummy_list_vocab_for_testing
):
    countvec = get_CountVecFeaturizer_instance_for_testing
    countvec.featurizer = get_CountVectorizer_instance_for_testing
    countvec._trainset = get_dummy_list_vocab_for_testing
    
    returned_vocab = countvec._get_vocab_from_trainset()
    
    assert isinstance(returned_vocab, set)


def test__update_loaded_vocabulary_method_expected_vocab_dict_obj_type_updated_successfully(
    get_CountVecFeaturizer_instance_for_testing,
    mocker
):
    vocab_from_trainset = ['red', 'blue']
    vocab_from_featurizer = {0:'orange', 1:'black'}
    
    countvec = get_CountVecFeaturizer_instance_for_testing

    mocker.patch.object(
        countvec, "_get_vocab_from_trainset",
        return_value=vocab_from_trainset
    )
    
    mocker.patch.object(
        countvec, "_get_vocab_from_featurizer",
        return_value=vocab_from_featurizer
    )
    
    updated_vocab = countvec._update_loaded_vocabulary()

    expected_vocab = {0: 'orange', 1: 'black', 'red': 2, 'blue': 3}

    assert isinstance(updated_vocab, dict)
    assert updated_vocab == expected_vocab


def test__train_method_expected_call_fit_with_correct_parameters(
    get_CountVecFeaturizer_instance_for_testing,
    get_CountVectorizer_instance_for_testing,
    get_dummy_list_vocab_for_testing,
    get_mocked_logger_at_info_level,
    mocker
):
    mocked_info_logger = get_mocked_logger_at_info_level
    
    trainset = get_dummy_list_vocab_for_testing
    
    countvec = get_CountVecFeaturizer_instance_for_testing
    countvec.featurizer = get_CountVectorizer_instance_for_testing
    countvec._trainset = trainset

    mocked_fit_method = mocker.patch.object(
        countvec.featurizer, 
        "fit"
    )
    
    countvec._train()
    
    mocked_fit_method.assert_called_once_with(trainset)
    mocked_info_logger.assert_any_call("'CountVecFeaturizer' training has started")
    mocked_info_logger.assert_any_call("'CountVecFeaturizer' training finished")
    
    
def test__train_loaded_featurizer_method_expected_call__train(
    get_CountVecFeaturizer_instance_for_testing,
    mocker
):
    countvec = get_CountVecFeaturizer_instance_for_testing
    
    mocked__train_method = mocker.patch.object(
        countvec, 
        "_train"
    )
    
    countvec._train_loaded_featurizer()
    
    mocked__train_method.assert_called_once()
    

def test__train_featurizer_from_scratch_expected_call___correct_methods(
    get_CountVecFeaturizer_instance_for_testing,
    mocker
):
    countvec = get_CountVecFeaturizer_instance_for_testing
    
    mocked__load_featurizer_params_method = mocker.patch.object(
        countvec, 
        "_load_featurizer_params"
    )
    
    mocked__create_featurizer_method = mocker.patch.object(
        countvec, 
        "_create_featurizer"
    )
    
    mocked__train_method = mocker.patch.object(
        countvec, 
        "_train"
    )
    
    countvec._train_featurizer_from_scratch()
    
    mocked__load_featurizer_params_method.assert_called_once()
    mocked__create_featurizer_method.assert_called_once()
    mocked__train_method.assert_called_once()
    
    
def test__train_featurizer_from_scratch_expected_featurizer_as_CountVectorizer_obj_type(
    get_CountVecFeaturizer_instance_for_testing,
    get_CountVectorizer_instance_for_testing,
    mocker
):
    countvec = get_CountVecFeaturizer_instance_for_testing
        
    mocker.patch.object(
        countvec, 
        "_create_featurizer",
        return_value=get_CountVectorizer_instance_for_testing
    )
    
    mocker.patch.object(
        countvec, 
        "_train"
    )
    
    countvec._train_featurizer_from_scratch()
    
    assert isinstance(countvec.featurizer, CountVectorizer)
    

def test_train_method_when_trainset_is_given_expected_set__trainset_correctly(
    get_CountVecFeaturizer_instance_for_testing,
    get_dummy_list_vocab_for_testing
):
    countvec = get_CountVecFeaturizer_instance_for_testing
    
    countvec.train(trainset=get_dummy_list_vocab_for_testing)
    
    assert countvec._trainset == get_dummy_list_vocab_for_testing
    
    
def test_train_method_when_featurizer_is_None_and__check_if_trained_featurizer_exists_and_load_it_method_is_True_expected_correct_method_calls(
    get_CountVecFeaturizer_instance_for_testing,
    get_dummy_list_vocab_for_testing,
    mocker
):
    countvec = get_CountVecFeaturizer_instance_for_testing
    
    countvec.featurizer = None
    
    mocked__check_if_trained_featurizer_exists_and_load_it_method = mocker.patch.object(
        countvec, 
        "_check_if_trained_featurizer_exists_and_load_it",
        return_value=False
    )
    
    mocked__train_featurizer_from_scratch_method = mocker.patch.object(
            countvec, 
            "_train_featurizer_from_scratch"
    )

    countvec.train(trainset=get_dummy_list_vocab_for_testing)
    
    mocked__check_if_trained_featurizer_exists_and_load_it_method.assert_called_once()
    mocked__train_featurizer_from_scratch_method.assert_called_once()
    

def test_train_method_when_featurizer_is_None_and__check_if_trained_featurizer_exists_and_load_it_method_is_False_expected_correct_method_calls(
    get_CountVecFeaturizer_instance_for_testing,
    get_dummy_list_vocab_for_testing,
    mocker
):
    countvec = get_CountVecFeaturizer_instance_for_testing
    
    countvec.featurizer = None
    
    mocked__check_if_trained_featurizer_exists_and_load_it_method = mocker.patch.object(
        countvec, 
        "_check_if_trained_featurizer_exists_and_load_it",
        return_value=True
    )
    
    mocked__train_loaded_featurizer_method = mocker.patch.object(
            countvec, 
            "_train_loaded_featurizer"
    )

    countvec.train(trainset=get_dummy_list_vocab_for_testing)
    
    mocked__check_if_trained_featurizer_exists_and_load_it_method.assert_called_once()
    mocked__train_loaded_featurizer_method.assert_called_once()
    

def test_train_method_when_featurizer_is_not_None_expected_call__train_loaded_featurizer_method(
    get_CountVecFeaturizer_instance_for_testing,
    get_dummy_list_vocab_for_testing,
    mocker
):
    countvec = get_CountVecFeaturizer_instance_for_testing
    
    countvec.featurizer = "loaded_vectorizer"
    
    mocked__train_loaded_featurizer_method = mocker.patch.object(
        countvec, 
        "_train_loaded_featurizer"
    )
    
    countvec.train(trainset=get_dummy_list_vocab_for_testing)
    
    mocked__train_loaded_featurizer_method.assert_called_once()


def test_load_method_when_featurizer_is_not_None_expected_set_featurizer_attr_correctly(
    get_CountVecFeaturizer_instance_for_testing,
    get_CountVectorizer_instance_for_testing
):
    countvec = get_CountVecFeaturizer_instance_for_testing
    
    countvec.load(featurizer=get_CountVectorizer_instance_for_testing)
    
    assert isinstance(countvec.featurizer, CountVectorizer)
    

def test_load_method_when_featurizer_is_None_expected_call__check_if_trained_featurizer_exists_and_load_it(
    get_CountVecFeaturizer_instance_for_testing,
    mocker
):
    countvec = get_CountVecFeaturizer_instance_for_testing
    
    mocked__check_if_trained_featurizer_exists_and_load_it_method = mocker.patch.object(
        countvec, 
        "_check_if_trained_featurizer_exists_and_load_it"
    )
    
    countvec.load(featurizer=None)
    
    mocked__check_if_trained_featurizer_exists_and_load_it_method.assert_called_once()


def test__persist_model_method_expected_call_save_data_from_callable_with_correct_params(
    get_CountVecFeaturizer_instance_for_testing,
    mocker
):
    countvec = get_CountVecFeaturizer_instance_for_testing
    
    model = "model"
    
    temp_pkl_file = utils.create_temp_pickle_file(
        obj=CountVectorizer(),
        suffix='.pkl',
        delete=False
    )

    countvec.path_to_save_model = temp_pkl_file

    mocked_save_data_from_callable_method = mocker.patch.object(
        TextFeaturizer.data_manager, 
        "save_data_from_callable"
    )

    mocked_persist_data_with_pickle_fn = mocker.patch(
        "pypipe.core.processes.utils.persist_data_with_pickle"
    )
    
    countvec._persist_model(model=model)
    
    mocked_save_data_from_callable_method.assert_called_once_with(
        model,
        "wb",
        callback_fn_to_save_data=mocked_persist_data_with_pickle_fn,
        path_to_save_data=temp_pkl_file,
        data_file_name="/vocabularies.pkl",
        alias="countvec_featurizer"
    )
    
    os.remove(temp_pkl_file)
    
    
def test__persist_vocab_method_expected_call_save_data_from_callable_with_correct_params(
    get_CountVecFeaturizer_instance_for_testing,
    get_dummy_dict_vocab_for_testing,
    mocker
):
    countvec = get_CountVecFeaturizer_instance_for_testing
    
    vocab = "vocab"
    
    temp_json_file = utils.create_temp_json_file_from_dict(
        obj=get_dummy_dict_vocab_for_testing,
        delete=False
    )

    countvec.path_to_save_vocabulary = temp_json_file

    mocked_save_data_from_callable_method = mocker.patch.object(
        TextFeaturizer.data_manager, 
        "save_data_from_callable"
    )

    mocked_persist_dict_as_json_fn = mocker.patch(
        "pypipe.core.processes.utils.persist_dict_as_json"
    )
    
    countvec._persist_vocab(vocab=vocab)
    
    mocked_save_data_from_callable_method.assert_called_once_with(
        vocab,
        "w",
        callback_fn_to_save_data=mocked_persist_dict_as_json_fn,
        path_to_save_data=temp_json_file,
        data_file_name="/vocab.json",
        alias="countvec_featurizer",
        to_save_vocab=True
    )
    
    os.remove(temp_json_file)
    

def test_persist_method_when_model_is_True_expected_call__persist_model_with_correct_param(
    get_CountVecFeaturizer_instance_for_testing,
    mocker
):
    countvec = get_CountVecFeaturizer_instance_for_testing
    
    mocked__persist_model_method = mocker.patch.object(
        countvec,
        "_persist_model"
    )
    
    countvec.persist(model=True, vocab=False)
    
    mocked__persist_model_method.assert_called_once_with(
        model=countvec.featurizer
    )
    
    
def test_persist_method_when_vocab_and__update_stored_vocabulary_are_True_and_selfvocab_is_not_None_expected_call_correct_methods(
    get_CountVecFeaturizer_instance_for_testing,
    get_dummy_dict_vocab_for_testing,
    mocker
):
    countvec = get_CountVecFeaturizer_instance_for_testing
    
    countvec._update_stored_vocabulary = True
    countvec.vocab = "vocab"
    
    mocked__update_loaded_vocabulary_method = mocker.patch.object(
        countvec,
        "_update_loaded_vocabulary",
        return_value=get_dummy_dict_vocab_for_testing
    )
    
    mocked__persist_vocab_method = mocker.patch.object(
        countvec,
        "_persist_vocab"
    )
    
    countvec.persist(model=False, vocab=True)
    
    mocked__update_loaded_vocabulary_method.assert_called_once()
    mocked__persist_vocab_method.assert_called_once_with(vocab=get_dummy_dict_vocab_for_testing)
    
    
def test_persist_method_when_vocab_and__update_stored_vocabulary_are_True_and_selfvocab_is_None_expected_warning_log(
    get_CountVecFeaturizer_instance_for_testing,
    get_mocked_logger_at_warning_level
):
    mocked_warning_logger = get_mocked_logger_at_warning_level    

    countvec = get_CountVecFeaturizer_instance_for_testing

    countvec._update_stored_vocabulary = True
    countvec.vocab = None
        
    countvec.persist(model=False, vocab=True)
    
    mocked_warning_logger.assert_any_call(
        f"Could not update stored vocabulary for 'CountVecFeaturizer' "
        f"'CountVecFeaturizer' because no stored vocabulary was found "
        f"in '{countvec._path_to_get_stored_vocabulary}'"
    )
    
    
def test_persist_method_when_vocab_is_True_and__update_stored_vocabulary_is_False_expected_call_correct_methods(
    get_CountVecFeaturizer_instance_for_testing,
    get_dummy_dict_vocab_for_testing,
    mocker
):
    countvec = get_CountVecFeaturizer_instance_for_testing
    
    countvec._update_stored_vocabulary = False
    
    mocked__get_vocab_from_featurizer_method = mocker.patch.object(
        countvec,
        "_get_vocab_from_featurizer",
        return_value=get_dummy_dict_vocab_for_testing
    )
    
    mocked__persist_vocab_method = mocker.patch.object(
        countvec,
        "_persist_vocab"
    )
    
    countvec.persist(model=False, vocab=True)
    
    mocked__get_vocab_from_featurizer_method.assert_called_once()
    mocked__persist_vocab_method.assert_called_once_with(vocab=get_dummy_dict_vocab_for_testing)
    

def test_process_method_when_corpus_is_given_and_featurizer_is_None_expected_correct_warning_log_and_corpus_returned(
    get_CountVecFeaturizer_instance_for_testing,
    get_dummy_list_vocab_for_testing,
    get_mocked_logger_at_warning_level
):
    mocked_warning_logger = get_mocked_logger_at_warning_level    

    countvec = get_CountVecFeaturizer_instance_for_testing
    
    countvec.featurizer = None
    
    x = countvec.process(get_dummy_list_vocab_for_testing)
    
    assert x == get_dummy_list_vocab_for_testing
    mocked_warning_logger.assert_any_call(
        "It's impossible to process the input from 'CountVecFeaturizer' "
        "because there is no trained model"
    )
    
    
def test_process_method_when_corpus_is_given_and_featurizer_is_not_None_expected_call_transform(
    get_CountVecFeaturizer_instance_for_testing,
    get_CountVectorizer_instance_for_testing,
    get_dummy_list_vocab_for_testing,
    mocker
):   
    import numpy as np
    
    countvec = get_CountVecFeaturizer_instance_for_testing
    
    countvec.featurizer = get_CountVectorizer_instance_for_testing
    
    mocked_transform_method = mocker.patch.object(
        countvec.featurizer,
        "transform",
        return_value=np.array([[0, 0, 1], [1, 0, 1]])
    )
    
    x = countvec.process(get_dummy_list_vocab_for_testing)
    
    assert isinstance(x, np.ndarray)
    mocked_transform_method.assert_called_once_with(get_dummy_list_vocab_for_testing)

