import os

from pypipe.api import create_app
from pypipe.api.core import blueprint


app = create_app(os.getenv("BOILERPLATE_ENV") or "dev")
app.register_blueprint(blueprint)

app.app_context().push()

def run():
    app.run()


if __name__ == '__main__':
    run()
    
