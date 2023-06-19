from flask import Flask
from flask_bcrypt import Bcrypt

from pypipe.api import config


flask_bcrypt = Bcrypt()


def create_app(config_name: str) -> Flask:  
    """
    Create and configure the Flask application.

    Parameters:
    - config_name (str): The name of the configuration to use for the 
        application.

    Returns:
    - Flask: The Flask application object.

    Description:
    This function creates a new Flask application object and configures it 
    based on the specified 'config_name'. It loads the configuration settings 
    from the 'config.py' module.

    Example usage:
    >> app = create_app('development')
    """
    app = Flask(__name__)
    app.config.from_object(config.get_config_by_name[config_name])
    flask_bcrypt.init_app(app)
    return app