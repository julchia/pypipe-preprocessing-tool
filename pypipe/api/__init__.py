from flask import Flask
from flask_bcrypt import Bcrypt

from pypipe.api import config


flask_bcrypt = Bcrypt()


def create_app(config_name: str) -> Flask:  
    """
    """
    app = Flask(__name__)
    app.config.from_object(config.get_config_by_name[config_name])
    flask_bcrypt.init_app(app)
    return app