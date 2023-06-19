from typing import Union, Iterable

from flask import Response, jsonify

from pypipe.core.pipeline.pipeline import Pipeline


def get_output_from_specific_pipeline_process(
    config: str,
    data: Union[Iterable[str], str],
    process_alias: str,
    persist: bool = False
) -> Response:
    """
    Get the output from a specific process within a pipeline and return it as 
    a JSON response.

    Parameters:
    - config: The path to the configuration file for the pipeline.
    - data: The input data to be processed by the pipeline.
    - process_alias: The alias of the specific process to retrieve the output from.
    - persist: Flag indicating whether to persist the output of the process.

    Returns:
    - Response: A Flask response object containing the JSON response.
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
    Get the output from all processes within a pipeline and return it as 
    a JSON response.

    Parameters:
    - config: The path to the configuration file for the pipeline.
    - data: The input data to be processed by the pipeline.
    - persist: Flag indicating whether to persist the output of the process.

    Returns:
    - Response: A Flask response object containing the JSON response.
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
    
    