from app import create_app
from gevent import pywsgi
from config import Config
from gevent import monkey


monkey.patch_all() # 多线程支持

app = create_app()
 
if __name__ == '__main__':
    http_server = pywsgi.WSGIServer((Config.WSGI_HOST, Config.WSGI_PORT), app)
    http_server.serve_forever()
