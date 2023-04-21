from flask.json import jsonify
from app.api import bp
from app import mqtt_client, redis_client
from app.data.parameters import Constant
from concurrent.futures import ThreadPoolExecutor


pool = ThreadPoolExecutor(max_workers=Constant.API_POOL_NUM)


def publish(unit):
    for key in Constant.REDIS_KEYS:
        data = redis_client.hget(unit, key)
        mqtt_client.publish(topic=f"{unit}_{key}", payload=data, qos=Constant.MQTT_QUALITY_OF_SERVICE)


@bp.route(f"/<any{Constant.API_OPTIONS}:unit>", methods=["GET"])
def cylindrical_valves(unit):
    pool.submit(publish, unit) # 实现相同API并发
    return jsonify("OK")
