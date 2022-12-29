from app.main import bp
from flask.json import jsonify


@bp.route("/health", methods=["GET", "POST"])
def health():
    return jsonify('The cylindrical valves intelligent control algorithm is checked OK.')
