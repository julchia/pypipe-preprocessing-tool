from flask_restx import Api
from flask import Blueprint

from pypipe.api.core.controller.pipeline_controllers import api as pipeline_ns


blueprint = Blueprint('api', __name__)

api = Api(blueprint,
          title='NLP PREPROCESSING PIPELINE API',
          version='1.0',
          description='Temporal boilerplate for preprocessing nlp service'
          )

api.add_namespace(pipeline_ns, path='/preprocessing_pipeline')