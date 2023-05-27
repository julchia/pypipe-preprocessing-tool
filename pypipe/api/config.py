import os


basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    """
    Base config factory
    
    Useful pattern to create multiple instances of our application with 
    different settings.
    """
    SECRET_KEY = os.getenv("SECRET_KEY", "tmp_key")
    DEBUG = False
    
    
class DevelopmentConfig(Config):
    """
    Concrete dev-config object
    """
    DEBUG = True
    
    
get_config_by_name = dict(
    dev=DevelopmentConfig
)


key = Config.SECRET_KEY
