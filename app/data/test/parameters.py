class Constant(object):
    DEFAULT_INTERVAL = 12
    MIN_CLOSE_TIME = 50
    MIN_OPEN_TIME = 50
    TIMER_DATA_LIMIT = 100
    RETRO_DATA_LIMIT = 100
    OVERTIME_DATA_INTERVAL_DAYS = 30
    RELAY_DISPLACEMENT_SAMPLING_POINTS = 100
    RELAY_DISPLACEMENT_TRIM_POINTS = 5
    EPSILON = 1e-8
    ACTION_TIME_SV_BEFORE_CV = 30
    MIN_PUMP_TIME = 10
    HEALTH_DATA_INTERVAL_DAYS = 1
    HEALTH_MS_DATA_SAMPLING_HZ = 600
    HEALTH_S_DATA_SAMPLING_HZ = 60
    REDIS_KEYS = [
        "close",
        "open",
        "pump1",
        "pump2",
        "pump3",
        "overtime",
        "retro_close",
        "retro_open",
        "health",
        "fit_close",
        "fit_open",
    ]
    API_POOL_NUM = 12
    GET_HEALTH_DF_POOL_NUM = 8
    HEALTH_POO_NUM = 6
    OPEN_CLOSE_POOL_NUM = 4
    GOF_POOL_NUM = 4