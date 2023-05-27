from flask import request
from flask_restplus import Resource

from pypipe.api.core.utils import dto
from pypipe.api.core.service.pipeline_services import (
    get_output_from_pipeline_processes_sequence, 
    get_output_from_specific_pipeline_process
)


api = dto.PipelineDTO.api
_pipeline = dto.PipelineDTO.pipeline


api.route('/pipeline_processing')
class PipelineSequenceResource(Resource):
    @api.response(200, 'Data processed correctly.')
    @api.doc('Processes text data by sequentially executing the processes set in the pipeline configuration.')
    @api.expect(_pipeline, validate=True)
    def post(self):
        """Processes text data by sequentially executing the processes set in 
        the pipeline configuration."""
        data = request.json
        return get_output_from_pipeline_processes_sequence(
            config=data.get('config'),
            data=data.get('data'),
            persist=data.get('persist')
        )


api.route('/pipeline_processing_by_specific_process')
class PipelineProcessResource(Resource):
    @api.response(200, 'Data processed correctly.')
    @api.doc('Processes text data by specific process set in the pipeline configuration.')
    @api.expect(_pipeline, validate=True)
    def post(self):
        """Processes text data by specific process set in the pipeline 
        configuration.""" 
        data = request.json
        return get_output_from_specific_pipeline_process(
            config=data.get('config'),
            data=data.get('data'),
            process_alias=data.get('process_alias'),
            persist=data.get('persist')
        )

