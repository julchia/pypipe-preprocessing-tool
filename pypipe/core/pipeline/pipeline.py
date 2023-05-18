from __future__ import annotations
from typing import List, Dict, Any, Union, Callable, Iterable

import logging
from omegaconf import OmegaConf, DictConfig

from pypipe.configs import config_const
from pypipe.core.pipeline import pipeline_const
from pypipe.core.interfaces import IProcess
from pypipe.core.management.managers import CorpusLazyManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    """
    This class represents a pipeline object used for 
    processing textual data through a set of sequential 
    processes.
    """
    _config_alias: Dict[str, str] = config_const.CONFIG_ALIAS
    _pipeline_process: Dict[str, Any] = pipeline_const.PIPELINE_PROCESS_ALIAS
    
    def __init__(
        self,
        config: str,
        corpus: Union[Iterable[str], str] = None,
        corpus_generator: Callable[[Union[List[str], str]], Iterable] = CorpusLazyManager
    ) -> None:
        """
        Builds a Pipeline object.
        
        Args:
            config: The alias of configuration file or path to configuration 
                file.

            corpus: The corpus to be processed. Can be a list
                of str or a path to static corpus file. 

            pipeline_process: A dictionary containing the alias and
                specification of each process in the pipeline. The
                specifications are a tuple with: 0) a PipeHandler object 
                and 1) a Processor object.
                
            corpus_generator: A callable object that returns an iterable
                 corpus
        """
        self._config = self._format_config(config=config)
        self._corpus_generator = corpus_generator
        if corpus is not None:
            self._corpus = self._corpus_generator(corpus)
    
    @staticmethod
    def _format_config(config: str) -> DictConfig: 
        """
        Load and return the configuration object.

        Args:
            config: The path to the configuration file.
        """
        if config in Pipeline._config_alias:
            config_path = Pipeline._config_alias[config]
            return OmegaConf.load(config_path)
        else:
            try:
                return OmegaConf.load(config)
            except FileNotFoundError:
                raise KeyError(
                    f"{config} is not a configuration alias."
                )
            
    def create_pipeline_process(self, alias: str) -> IProcess:
        """Returns a pipeline process."""
        if alias in Pipeline._pipeline_process:
            if self._config.pipeline[alias].active:
                _, processor = Pipeline._pipeline_process[alias]
                process = processor(alias=alias, configs=self._config)
                self._pipiline_was_created = True
                return process
                   
    def run_processes_sequentially(
        self, 
        corpus: Union[Iterable[str], str] = None,
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
        if corpus is not None:
            self._corpus = self._corpus_generator(corpus)
        
        processed_corpus = self._corpus
        
        for alias, spec in Pipeline._pipeline_process.items():
            if self._config.pipeline[alias].active:
                process = self.create_pipeline_process(alias)
                handler, _ = spec
                active_handler = handler(processor=process)
                processed_corpus = active_handler.process(
                    corpus=processed_corpus,
                    persist=persist
                )   
                
        return processed_corpus
            
