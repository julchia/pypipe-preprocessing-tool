from __future__ import annotations
from typing import List, Dict, Any, Union, Callable, Iterable

import logging
from omegaconf import OmegaConf, DictConfig

from pypipe import settings
from pypipe.core.pipeline import constants
from pypipe.core.interfaces import IProcess
from pypipe.core.management.managers import DataLazyManager


logging.basicConfig(filename=settings.LOG_DIR,  level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

class Pipeline:
    """
    This class represents a pipeline object used for 
    processing textual data through a set of sequential 
    processes.
    """
    _config_alias: Dict[str, str] = settings.CONFIG_ALIAS
    _pipeline_process: Dict[str, Any] = constants.PIPELINE_PROCESS_ALIAS
    
    def __init__(
        self,
        config: str,
        data: Union[Iterable[str], str] = None,
        data_generator: Callable[[Union[List[str], str]], Iterable] = DataLazyManager
    ) -> None:
        """
        Builds a Pipeline object.
        
        Args:
            config: The alias of configuration file or path to configuration 
                file.

            data: The data to be processed. Can be a list
                of str or a path to static data corpus file. 

            pipeline_process: A dictionary containing the alias and
                specification of each process in the pipeline. The
                specifications are a tuple with: 0) a PipeHandler object 
                and 1) a Processor object.
                
            data_generator: A callable object that returns an iterable
                 data corpus
        """
        self._config = self._format_config(config=config)
        self._data_generator = data_generator
        if data is not None:
            self._data = self._data_generator(data)
    
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
        data: Union[Iterable[str], str] = None,
        persist: bool = False,
    ) -> Union[Iterable[str], str]:
        """
        Process the data corpus through the pipeline in sequential 
        order.

        Args:
            data: The data corpus to be processed. Can be a list
            of str or a path to static data corpus file.  
            
            persist: If there are paths set in the configurations, persists
            all outputs of all processes in the executed sequence.
        """
        if data is not None:
            self._data = self._data_generator(data)
        
        processed_data = self._data
        
        for alias, spec in Pipeline._pipeline_process.items():
            if self._config.pipeline[alias].active:
                process = self.create_pipeline_process(alias)
                handler, _ = spec
                active_handler = handler(processor=process)
                processed_data = active_handler.process(
                    data=processed_data,
                    persist=persist
                )   
                
        return processed_data
            
