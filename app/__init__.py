from flask import Flask
from config import Config
from flask_mqtt import Mqtt
from flask_apscheduler import APScheduler as _BaseAPScheduler
from flask_redis import FlaskRedis
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler
import os
import warnings
warnings.filterwarnings("ignore")


# 重写APScheduler,如果不重写APScheduler类,则定时任务函数涉及上下文操作时,app必须与定时任务函数在一个模块内
class APScheduler(_BaseAPScheduler):
    def run_job(self, id, jobstore=None):
        with self.app.app_context():
            super().run_job(id=id, jobstore=jobstore)

scheduler = APScheduler()


def scheduler_start(app):
    """ 多进程进行部署，定时任务会重复启动 """
    import platform
    import atexit

    if platform.system() != 'Windows':
        fcntl = __import__("fcntl")
        f = open('scheduler.lock', 'wb')
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            scheduler.start()
            app.logger.debug('Scheduler Started,---------------')
        except:
            pass
 
        def unlock():
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()

        atexit.register(unlock)
    else:
        msvcrt = __import__('msvcrt')
        f = open('scheduler.lock', 'wb')
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            scheduler.start()
            app.logger.debug('Scheduler Started,----------------')
        except:
            pass
 
        def _unlock_file():
            try:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except:
                pass
 
        atexit.register(_unlock_file)

mqtt_client = Mqtt()
redis_client = FlaskRedis()
cors = CORS()


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    cors.init_app(app, supports_credentials=True) # 允许跨域请求

    scheduler.init_app(app)
    scheduler_start(app)

    mqtt_client.init_app(app)
    redis_client.init_app(app)

    from app.main import bp as main_bp
    app.register_blueprint(main_bp)

    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    if not app.debug and not app.testing:
        if not os.path.exists(Config.LOG_FILE_PATH):
            os.mkdir(Config.LOG_FILE_PATH)
        file_handler = RotatingFileHandler(
            Config.LOG_FILE_PATH + Config.LOG_FILE_NAME,
            maxBytes=Config.LOG_FILE_SIZE,
            backupCount=Config.LOG_FILE_SUM
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

        app.logger.setLevel(logging.INFO)
        app.logger.info(Config.LOG_FILE_START)   

    return app