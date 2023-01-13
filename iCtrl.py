from app import create_app, logger
from gevent import pywsgi, monkey
from config import Config


monkey.patch_all() # 多线程支持

app = create_app()
 
if __name__ == '__main__':
    with logger.catch():
        try:
            http_server = pywsgi.WSGIServer((Config.WSGI_HOST, Config.WSGI_PORT), app)
            http_server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Exit by KeyboardInterrupt")