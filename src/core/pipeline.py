from typing import Dict

from omegaconf import OmegaConf

from src.core import constants
from src.core.interfaces import IProcessHandler, IPipelineProcess


class Pipeline:
    
    def __init__(
        self, 
        pipeline_conf: OmegaConf, 
        pipeline_process: Dict[str, IPipelineProcess] = constants.PIPELINE_PROCESSES_ALIAS
    ) -> None:
        self._pipeline_conf = pipeline_conf
        self._pipeline_process = pipeline_process
        self.__init_pipe()
    
    def __init_pipe(self):
        self._pipeline: Dict[str, IProcessHandler] = {}
        self._get_pipeline_processes()
        self._init_pipe_processes()
        return self
    
    def _init_pipe_processes(self):
        self._init_regex_normalization_process()
        self._init_featurization_process()
    
    def _init_regex_normalization_process(self):
        self.regex_normalization = self._pipeline.get("regex_normalization")
        
    def _init_featurization_process(self):
        self.featurization = self._pipeline.get("featurization")
    
    def _get_pipeline_processes(self):
        for k, process in self._pipeline_process.items():
            if k in self._pipeline_conf.pipeline:
                self._pipeline[k] = process(self._pipeline_conf).get_process()
