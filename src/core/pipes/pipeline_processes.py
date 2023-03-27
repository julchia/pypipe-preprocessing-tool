from omegaconf import OmegaConf

from src.core.interfaces import IProcessHandler, IPipelineProcess
from src.core.builders.process_builders import ProcessBuilder
from src.core.handlers import regex_handlers, featurizer_handlers


class RegexNormalizationProcess(IPipelineProcess):
    
    def __init__(self, pipeline_conf: OmegaConf) -> None:
        super().__init__(
            pipeline_conf=pipeline_conf,
            process_builder=ProcessBuilder(),
            process_handlers=regex_handlers.__dict__
        )       
    
    def _build_process_sequence(self) -> IProcessHandler:
        for handler, configs in self._pipeline_conf.pipeline.regex_normalization.items():
            if configs.active:
                self.add_step_to_sequence(
                    handler_name=handler,
                    handler_configs=configs
                )     
        return self._process_builder._build_process()
    
    def get_process(self) -> IProcessHandler:
        return self._build_process_sequence()


class FeaturizationProcess(IPipelineProcess):
    
    def __init__(self, pipeline_conf: OmegaConf) -> None:
        super().__init__(
            pipeline_conf=pipeline_conf,
            process_builder=ProcessBuilder(),
            process_handlers=featurizer_handlers.__dict__
        )       
    
    def _build_process_sequence(self) -> IProcessHandler:
        for handler, configs in self._pipeline_conf.pipeline.featurization.items():
            if configs.active:
                self.add_step_to_sequence(
                    handler_name=handler,
                    handler_configs=configs
                )     
        return self._process_builder._build_process()
    
    def get_process(self) -> IProcessHandler:
        return self._build_process_sequence()