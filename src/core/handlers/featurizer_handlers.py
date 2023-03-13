from omegaconf import OmegaConf

from src.core.interfaces import IProcessHandler
from src.core.handlers.process_handlers import ProcessHandler


class TextFeaturizer(ProcessHandler):
    """
    """
    
    def __init__(self, configs: OmegaConf, next_processor: IProcessHandler = None) -> None:
        super().__init__(next_processor)
        self._configs = configs
    
    def featurize_text(self):
        print("texto a features")


class SklearnTfIdfFeaturizer(TextFeaturizer):
    """
    """
    
    def _process(self):
        print("texto a tfidf-features")
        
    
class Word2VecFeaturizer(TextFeaturizer):
    """
    """
    
    def _process(self):
        print("texto a word2vec-features")
