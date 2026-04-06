"""
HTTP API views — mirrors views/qa_apis.py from archit1012/qa-bot-llm.

Creates shared MistralConfig once (like global ChatOpenAI in the reference) and
injects it into the QA service for each /upload call.
"""
import os

from flask import Blueprint, jsonify, request

from ..service import qa_service

bp = Blueprint("qa", __name__)

_llm_config = None


def _get_llm_config():
    global _llm_config
    if _llm_config is None:
        _llm_config = qa_service.mistral_config_from_env()
    return _llm_config


@bp.route("/")
def hello_world():
    return "Hello World"


@bp.route("/upload", methods=["POST"])
def upload_files():
    try:
        cfg = _get_llm_config()
        if cfg is None:
            return jsonify({
                "error": "LLM not configured. Set MISTRAL_API_KEY and install langchain-mistralai.",
            }), 503

        embedding_model = os.getenv("EMBEDDING_MODEL")
        result, err = qa_service.process_qa_upload(
            request, cfg, embedding_model=embedding_model
        )
        if err:
            return jsonify({"error": err}), 400
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def register_routes(app):
    """Register QA blueprint on the Flask app."""
    app.register_blueprint(bp)
