from flask_restplus import Namespace, fields


class PipelineDTO:
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
    