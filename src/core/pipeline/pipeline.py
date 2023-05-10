from __future__ import annotations
from typing import List, Dict, Any, Union, Callable, Iterable

import logging
from omegaconf import OmegaConf, DictConfig

from src.core import constants
from src.core.interfaces import IProcess
from src.core.management.managers import CorpusLazyManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    """
    This class represents a pipeline object used for 
    processing textual data through a set of sequential 
    processes.
    """
    def __init__(
        self, 
        config_path: str,
        pipeline_process: Dict[str, Any] = constants.PIPELINE_PROCESS_ALIAS,
        corpus_generator: Callable[[Union[List[str], str]], Iterable] = CorpusLazyManager
    ) -> None:
        """
        Builds a Pipeline object.
        
        Args:
            config_path: The path to the configuration file.
            
            pipeline_process: A dictionary containing the alias and
                specification of each process in the pipeline. The
                specifications are a tuple with: 0) a PipeHandler object 
                and 1) a Processor object.
                
            corpus_generator: A callable object that returns an iterable
                 corpus
        """
        self._config = self._format_config(config=config_path)
        self._pipeline_process = pipeline_process
        self._corpus_generator = corpus_generator
    
    @staticmethod
    def _format_config(config: str) -> DictConfig: 
        """
        Load and return the configuration object.

        Args:
            config: The path to the configuration file.
        """
        return OmegaConf.load(config)
    
    def create_pipeline_process(self, alias: str) -> IProcess:
        """Returns a pipeline process."""
        if alias in self._pipeline_process:
            if self._config.pipeline[alias].active:
                _, processor = self._pipeline_process[alias]
                process = processor(alias=alias, configs=self._config)
                self._pipiline_was_created = True
                return process
                
    def run_processes_sequentially(
        self, 
        corpus: Union[Iterable[str], str],
        persist: bool = False,
    ) -> Union[Iterable[str], str]:
        """
        Process the corpus through the pipeline in sequential 
        order.

        Args:
            corpus: The corpus to be processed. Can be a list
            of str or a path to static corpus file.  
            
            persist: If there are paths set in the configurations, persists
            all outputs of all processes in the executed sequence.
        """
        processed_corpus = self._corpus_generator(corpus)
        for alias, spec in self._pipeline_process.items():
            if self._config.pipeline[alias].active:
                process = self.create_pipeline_process(alias)
                handler, _ = spec
                active_handler = handler(processor=process)
                processed_corpus = active_handler.process(
                    corpus=processed_corpus,
                    persist=persist
                )   
        return processed_corpus
            
