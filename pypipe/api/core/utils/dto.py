"""
This module provides Data transfer objects (DTOs) that define the data 
structures used for transferring data between different components of the 
application.
"""

from flask_restx import Namespace, fields


class PipelineDTO:
    """
    DTO for the Text Preprocessing Pipeline API.
    """
    api = Namespace(
        'preprocessing-pipeline', 
        description='Text Preprocessing Pipeline API'
    )
    pipeline = api.model('pipeline', {
        'config': fields.Raw(required=True, description='Pipeline configuration'),
        'data': fields.List(fields.String, required=True, description='Input data'),
        'process_alias': fields.String(required=False, description='Process alias identifier'),
        'persist': fields.Boolean(required=False, description='Persists the preprocessed data in the directory specified in config')
    })
    