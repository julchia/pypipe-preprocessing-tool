from __future__ import annotations
from typing import List, Dict

from omegaconf import OmegaConf

from src.core import constants
from src.core.interfaces import IPipelineProcess


class Pipeline:
    
    def __init__(
        self, 
        pipeline_conf: OmegaConf, 
        pipeline_process: Dict[str, IPipelineProcess] = constants.PIPELINE_PROCESSES_ALIAS
    ) -> None:
        self._pipeline_conf = pipeline_conf
        self._pipeline_process = pipeline_process
        self.__init_pipe()
    
    def __init_pipe(self) -> Pipeline:
        self._set_pipeline_processes()
        return self
    
    def _set_pipeline_processes(self) -> None:
        for k, process in self._pipeline_process.items():
            if k in self._pipeline_conf.pipeline:
                self.__dict__[k] = process(self._pipeline_conf).get_process()
                
    def get_processes_order(self) -> List:
        processes_order = list(self._pipeline_process.keys())
        return processes_order
