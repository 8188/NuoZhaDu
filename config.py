import os
from dotenv import load_dotenv
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore


basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

def task1_jobs(num):
    return {
        'id': f'task1_job{num}', 
        'func': 'app.data.data_make:task1', 
        'args': (num,), 
        'trigger': 'interval',
        'seconds': 10,
        'jitter': 1, # delay the job execution by ``jitter`` seconds at most
        'replace_existing': True, # 对于持久化任务,需要在启动时覆盖已存在的任务
    }


def task2_jobs(num):
    return {
        'id': f'task2_job{num}', 
        'func': 'app.data.data_make:task2', 
        'args': (num,), 
        'trigger': 'interval',
        'seconds': 60,
        'jitter': 5,
        'replace_existing': True, # 对于持久化任务,需要在启动时覆盖已存在的任务
    }


class Config(object):
    LOG_FILE_PATH = 'logs/'
    LOG_FILE_NAME = 'NUOZHADU.log'
    LOG_FILE_SIZE = 1024000 # 1000kb
    LOG_FILE_SUM = 10

    user = os.getenv('MYSQL_USER')
    password = os.getenv('MYSQL_PASSWORD')
    ip = os.getenv('MYSQL_IP')
    port = os.getenv('MYSQL_PORT')
    database = os.getenv('MYSQL_DATABASE')
    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{user}:{password}@{ip}:{port}/{database}"
    MYSQL_DATABASE_URI = f"mysql://{user}:{password}@{ip}:{port}/{database}"

    REDIS_HOST = os.getenv('REDIS_HOST')
    REDIS_PORT = os.getenv('REDIS_PORT')
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
    REDIS_DATABASE = os.getenv('REDIS_DATABASE')
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DATABASE}"

    MQTT_BROKER_URL = os.getenv('MQTT_BROKER_URL')
    MQTT_BROKER_PORT = int(os.getenv('MQTT_BROKER_PORT'))
    MQTT_USERNAME = os.getenv('MQTT_USERNAME')
    MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
    MQTT_KEEPALIVE = 5
    MQTT_TLS_ENABLED = False

    WSGI_HOST = os.getenv('WEB_SERVER_GATEWAY_INTERFACE_HOST')
    WSGI_PORT = int(os.getenv('WEB_SERVER_GATEWAY_INTERFACE_PORT'))

    # 调度开关开启
    SCHEDULER_API_ENABLED = True
    # 解决FLASK DEBUG模式定时任务执行两次
    WERKZEUG_RUN_MAIN = True
    # 持久化配置
    # SCHEDULER_JOBSTORES = {'default': SQLAlchemyJobStore(SQLALCHEMY_DATABASE_URI)}
    # 线程池配置，最大20个线程
    SCHEDULER_EXECUTORS = {
        'default': {'type': 'threadpool', 'max_workers': 20}
        }
    # 超过最大进程合并任务,最大进程,错过可执行时间,解决与gevent的猴子补丁冲突导致job miss
    SCHEDULER_JOB_DEFAULTS = {'coalesce': True, 'max_instances': 9, 'misfire_grace_time': 5}
    SCHEDULER_TIMEZONE = os.getenv('TIMEZONE')
    JOBS = [task1_jobs(i) for i in range(1, 10)] + [task2_jobs(i) for i in range(1, 10)]

    # https://ascii-generator.site/t/
    LOG_FILE_START = '''
          _____                    _____                    _____          
         /\    \                  /\    \                  /\    \         
        /::\    \                /::\    \                /::\    \        
       /::::\    \              /::::\    \              /::::\    \       
      /::::::\    \            /::::::\    \            /::::::\    \      
     /:::/\:::\    \          /:::/\:::\    \          /:::/\:::\    \     
    /:::/  \:::\    \        /:::/__\:::\    \        /:::/  \:::\    \    
   /:::/    \:::\    \      /::::\   \:::\    \      /:::/    \:::\    \   
  /:::/    / \:::\    \    /::::::\   \:::\    \    /:::/    / \:::\    \  
 /:::/    /   \:::\ ___\  /:::/\:::\   \:::\    \  /:::/    /   \:::\    \ 
/:::/____/     \:::|    |/:::/__\:::\   \:::\____\/:::/____/     \:::\____\\
\:::\    \     /:::|____|\:::\   \:::\   \::/    /\:::\    \      \::/    /
 \:::\    \   /:::/    /  \:::\   \:::\   \/____/  \:::\    \      \/____/ 
  \:::\    \ /:::/    /    \:::\   \:::\    \       \:::\    \             
   \:::\    /:::/    /      \:::\   \:::\____\       \:::\    \            
    \:::\  /:::/    /        \:::\   \::/    /        \:::\    \           
     \:::\/:::/    /          \:::\   \/____/          \:::\    \          
      \::::::/    /            \:::\    \               \:::\    \         
       \::::/    /              \:::\____\               \:::\____\        
        \::/    /                \::/    /                \::/    /        
         \/____/                  \/____/                  \/____/         
'''
