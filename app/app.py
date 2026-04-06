"""
Flask application entry — mirrors app/app.py from archit1012/qa-bot-llm.

Run from repository root::

    pip install -e .
    python -m app.app

Or::

    flask --app app.app:create_app run --debug --port 5000
"""
import os

from dotenv import load_dotenv
from flask import Flask

load_dotenv()


def create_app() -> Flask:
    app = Flask(__name__)
    from .views.qa_apis import register_routes

    register_routes(app)
    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=True, use_reloader=True, port=port)
