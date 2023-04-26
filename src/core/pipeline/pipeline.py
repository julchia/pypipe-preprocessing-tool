from __future__ import annotations
from typing import List, Dict, Any, Optional, Union, Callable, Iterable

import logging
from omegaconf import OmegaConf

from src.core import constants
from src.core.management.managers import CorpusLazyManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    """
    """
    def __init__(
        self, 
        pipeline_conf: OmegaConf, 
        pipeline_process: Dict[str, Any] = constants.PIPELINE_PROCESS_ALIAS,
        corpus_generator: Callable[[Union[List[str], str]], Iterable] = CorpusLazyManager
    ) -> None:
        self._pipeline_conf = pipeline_conf
        self._pipeline_process = pipeline_process
        self._corpus_generator = corpus_generator
        self._pipiline_was_created: bool = False
    
    def _set_pipeline_processes(self) -> None:
        """
        """
        for alias, spec in self._pipeline_process.items():
            if alias in self._pipeline_conf.pipeline:
                if self._pipeline_conf.pipeline[alias].active:
                    _, processor = spec
                    self.__dict__[alias] = processor(
                        alias=alias,
                        configs=self._pipeline_conf
                    )
            self._pipiline_was_created = True
        
    def _get_processes_sequentially(
        self, 
        corpus: Union[List[str], str]
    ) -> Optional[Any]:
        """
        """
        processed_corpus = self._corpus_generator(corpus)
        for alias, spec in self._pipeline_process.items():
            if alias in self.__dict__:
                handler, _ = spec
                active_handler = handler(
                    processor=self.__dict__[alias]
                )
                processed_corpus = active_handler.process(processed_corpus)
        return processed_corpus
           
    def create_pipeline(self) -> Pipeline:
        """
        """
        self._set_pipeline_processes()
        return self
    
    def process_corpus_sequentially(
        self, 
        corpus: Union[List[str], str]
    ) -> Optional[Any]:
        """
        """
        if not self._pipiline_was_created:
            logger.info(
                "To run processes sequentially, it is first necessary "
                "to create a pipeline by calling 'create_pipeline()'"
            )
        else:
            return self._get_processes_sequentially(corpus=corpus)
    
