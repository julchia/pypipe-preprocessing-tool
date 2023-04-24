from typing import List, Union, Iterable

from src.core.interfaces import IPipeHandler
from src.core.management.managers import CorpusLazyManager
from src.core.processes.normalization.normalizers import TextNormalizer
from src.core.processes.featurization.featurizers import TextFeaturizer


class PipeHandler(IPipeHandler):
    """
    """
    
    _next_handler: IPipeHandler = None
            
    @staticmethod
    def _check_if_input_type_is_iterable(
        iter_corpus: Iterable
    ) -> bool:
        if isinstance(iter_corpus, Iterable):
            return True
        else:
            raise ValueError(
                f"Invalid input type. Expected Iterable. "
                f"Received {type(iter_corpus)}"
            )
    
    def set_next(self, handler: IPipeHandler) -> IPipeHandler:
        self._next_handler = handler
        return handler


class TextNormalizerHandler(PipeHandler):
    """
    """ 
    def __init__(self, processor: TextNormalizer):
        self._processor = processor
       
    def process(self, corpus: Union[Iterable[str], List[str]]) -> CorpusLazyManager:
        if super()._check_if_input_type_is_iterable(iter_corpus=corpus):
            return self._processor.normalize_text(corpus=corpus)


class TextFeaturizerHandler(PipeHandler):
    """
    """
    def __init__(self, processor: TextFeaturizer):
        self._processor = processor
    
    def process(self, corpus: Union[Iterable[str], List[str]]) -> None:
        if super()._check_if_input_type_is_iterable(iter_corpus=corpus):                     
            return self._processor.train(trainset=corpus)

