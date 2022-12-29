from flask import Blueprint

bp = Blueprint('api', __name__)

from app.api import api

from flask_apscheduler import APScheduler

scheduler = APScheduler()