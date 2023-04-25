from __future__ import annotations
from typing import List, Dict, Any, Optional, Union, Callable, Iterable

from omegaconf import OmegaConf

from src.core import constants
from src.core.management.managers import CorpusLazyManager


class Pipeline:
    """
    """
    def __init__(
        self, 
        pipeline_conf: OmegaConf, 
        pipeline_process: Dict[str, Any] = constants.PIPELINE_PROCESS_ALIAS,
        corpus_iterator: Callable[[Union[List[str], str]], Iterable] = CorpusLazyManager
    ) -> None:
        self._pipeline_conf = pipeline_conf
        self._pipeline_process = pipeline_process
        self._corpus_iterator = corpus_iterator
    
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
                
    def create_pipeline(self) -> Pipeline:
        """
        """
        self._set_pipeline_processes()
        return self
    
    def process_corpus_sequentially(self, corpus: Union[List[str], str]) -> Optional[Any]:
        """
        """
        processed_corpus = self._corpus_iterator(corpus)
        
        for alias, spec in self._pipeline_process.items():
            if alias in self.__dict__:
                handler, _ = spec
                active_handler = handler(
                    processor=self.__dict__[alias]
                )
                processed_corpus = active_handler.process(processed_corpus)
    
        return processed_corpus
    
