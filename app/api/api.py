from app.api import bp
from app import mqtt_client
from app import redis_client
from app.data.parameters import Constant
from concurrent.futures import ThreadPoolExecutor
# from flask_cors import cross_origin


pool = ThreadPoolExecutor(max_workers=Constant.API_POOL_NUM)

def publish(unit):
    for key in Constant.REDIS_KEYS:
        data = redis_client.hget(unit, key)
        mqtt_client.publish(topic=f"{unit}_{key}", payload=data)


@bp.route('/<string:unit>', methods=['GET'])
# @cross_origin(supports_credentials=True)
def cylindrical_valves(unit):
    pool.submit(publish, unit) # 实现相同API并发
    return "OK"
