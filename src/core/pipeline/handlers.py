from typing import List, Union, Iterable

import logging

from src.core.interfaces import IProcessHandler
from src.core.management.managers import CorpusLazyManager
from src.core.processes.normalization.normalizers import TextNormalizer
from src.core.processes.featurization.featurizers import TextFeaturizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipeHandler(IProcessHandler):
    """
    """
    _prev_handler: IProcessHandler = None
    _next_handler: IProcessHandler = None
            
    @staticmethod
    def _check_if_input_type_is_iterable(
        iter_corpus: Iterable
    ) -> bool:
        """
        """
        if isinstance(iter_corpus, Iterable):
            return True
        else:
            raise ValueError(
                f"Invalid input type. Expected Iterable. "
                f"Received {type(iter_corpus)}"
            )
            
    def set_next(self, handler: IProcessHandler) -> IProcessHandler:
        """
        """
        self._next_handler = handler
        return handler


class TextNormalizerHandler(PipeHandler):
    """
    """ 
    def __init__(self, processor: TextNormalizer):
        self._processor = processor
       
    def process(self, corpus: Union[Iterable[str], List[str]]) -> CorpusLazyManager:
        """
        """
        PipeHandler._prev_handler = self
        if super()._check_if_input_type_is_iterable(iter_corpus=corpus):
            self._prev_handler = self
            return self._processor.normalize_text(corpus=corpus)


class TextFeaturizerHandler(PipeHandler):
    """
    """
    def __init__(self, processor: TextFeaturizer):
        self._processor = processor
    
    def process(self, corpus: Union[Iterable[str], List[str]]) -> None:
        """
        """       
        if not isinstance(self._prev_handler, TextFeaturizerHandler):
            PipeHandler._prev_handler = self
            if super()._check_if_input_type_is_iterable(iter_corpus=corpus):    
                return self._processor.train(trainset=corpus)
        else:
            logger.info(
                f"{self._processor} was not integrated into sequential execution "
                f"because it is preceded by another featurizer in the workflow"
            )
            return None
        