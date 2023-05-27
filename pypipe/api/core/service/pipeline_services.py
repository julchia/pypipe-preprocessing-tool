from typing import Union, Iterable

from flask import Response, jsonify

from pypipe.core.interfaces import IProcess
from pypipe.core.pipeline.pipeline import Pipeline


def get_output_from_specific_pipeline_process(
    config: str,
    data: Union[Iterable[str], str],
    process_alias: str,
    persist: bool = False
) -> Response:
    """
    """
    service = Pipeline(config=config)
    specific_process = service.create_pipeline_process(alias=process_alias)
    processed_data = specific_process.process(data=data, persist=persist)
    return jsonify({
        "status": "success",
        "message": "Data processed correctly",
        "processedData": processed_data
    })


def get_output_from_pipeline_processes_sequence(
    config: str,
    data: Union[Iterable[str], str],
    persist: bool = False
) -> Response:
    """
    """
    service = Pipeline(config=config)
    processed_data = service.run_processes_sequentially(
        data=data, 
        persist=persist
    )
    return jsonify({
        "status": "success",
        "message": "Data processed correctly",
        "processedData": processed_data
    })
    
    