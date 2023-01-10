from app.api import bp
from app import mqtt_client, redis_client, logger
from app.data.parameters import Constant
from concurrent.futures import ThreadPoolExecutor


pool = ThreadPoolExecutor(max_workers=Constant.API_POOL_NUM)

@logger.catch
def publish(unit):
    for key in Constant.REDIS_KEYS:
        data = redis_client.hget(unit, key)
        mqtt_client.publish(topic=f"{unit}_{key}", payload=data, qos=Constant.MQTT_QUALITY_OF_SERVICE)


@bp.route('/<string:unit>', methods=['GET'])
@logger.catch # 需放在@bp下方
def cylindrical_valves(unit):
    pool.submit(publish, unit) # 实现相同API并发
    return "OK"
