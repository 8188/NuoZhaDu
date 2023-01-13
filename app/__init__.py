from flask import Flask
from config import Config
from flask_mqtt import Mqtt
from flask_apscheduler import APScheduler as _BaseAPScheduler
from sqlalchemy import create_engine
from flask_redis import FlaskRedis
from flask_cors import CORS
from loguru import logger
import os
import warnings
warnings.filterwarnings("ignore")


# 重写APScheduler,如果不重写APScheduler类,则定时任务函数涉及上下文操作时,app必须与定时任务函数在一个模块内
class APScheduler(_BaseAPScheduler):
    def run_job(self, id, jobstore=None):
        with self.app.app_context():
            super().run_job(id=id, jobstore=jobstore)

scheduler = APScheduler()


def scheduler_lock():
    """ 多进程进行部署，定时任务会重复启动 
    参考: https://blog.csdn.net/u014595589/article/details/105083571
    """
    import platform
    import atexit

    if platform.system() != 'Windows':
        fcntl = __import__("fcntl")
        f = open('scheduler.lock', 'wb')
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            scheduler.start()
            logger.info("Scheduler Started")
        except:
            logger.error("Scheduler Failed to Start")
 
        def unlock():
            try:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
                logger.info("Lock Released")
                print("--------------------------Good Bye--------------------------")
            except:
                logger.error("Lock Failed to Release")

        atexit.register(unlock)
    else:
        msvcrt = __import__('msvcrt')
        f = open('scheduler.lock', 'wb')
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            scheduler.start()
            logger.info("Scheduler Started")
        except:
            logger.error("Scheduler Failed to Start")
 
        def _unlock_file():
            try:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                logger.info("Lock Released")
                print("--------------------------Good Bye--------------------------")
            except:
                logger.error("Lock Failed to Release")
 
        atexit.register(_unlock_file)

mqtt_client = Mqtt()
engine = create_engine(Config.MYSQL_DATABASE_URI)
redis_client = FlaskRedis()
cors = CORS()


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    cors.init_app(app, supports_credentials=True) # 允许跨域请求

    mqtt_client.init_app(app)
    redis_client.init_app(app)

    from app.main import bp as main_bp
    app.register_blueprint(main_bp)

    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    if not app.debug and not app.testing:
        if not os.path.exists(Config.LOG_FILE_PATH):
            os.mkdir(Config.LOG_FILE_PATH)

        logger.add(
            Config.LOG_FILE_PATH + Config.LOG_FILE_NAME,
            level=Config.LOG_LEVEL,
            rotation=Config.LOG_FILE_SIZE,
            retention=Config.LOG_FILE_SUM
        )

        print(Config.START_LOGO)
        logger.info("NUOZHADU MICROSERVER START")

    scheduler.init_app(app)
    scheduler_lock()

    return app