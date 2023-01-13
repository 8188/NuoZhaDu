from app.main import bp
from flask.json import jsonify
from app import logger
import os


@bp.route("/health", methods=["GET", "POST"])
def health():
    return jsonify('The cylindrical valves intelligent control algorithm is checked OK.')


@bp.route('/quit')
def _quit():
    logger.info("API Request to Exit")
    try:
        os._exit(0)
    except:
        logger.error("Exit error")