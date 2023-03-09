from src.core.handlers.process_handlers import ProcessHandler


class TextFeaturizer(ProcessHandler):
    """
    """
    
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
