from omegaconf import OmegaConf

from src.core import constants
from src.core.interfaces import IProcessHandler
from src.core.builders.process_builders import ProcessBuilder
from src.core.pipes.compound.compound_processor import CompoundProcessor


class RegexNormCompoundProcess(CompoundProcessor):
    
    def __init__(self, alias, configs: OmegaConf) -> None:
        super().__init__(
            alias=alias,
            configs=configs,
            process_builder=ProcessBuilder(),
            process_handlers=constants.COMPOUND_PROCESS_ALIAS[alias],
        )       
    
    def _build_process_sequence(self) -> IProcessHandler:
        for handler, configs in self._configs.items():
            if configs.active:
                self.add_step_to_sequence(
                    handler_name=handler,
                    handler_configs=configs
                )     
        return self._process_builder._build_process()
    
    def get_process(self) -> IProcessHandler:
        return self._build_process_sequence()