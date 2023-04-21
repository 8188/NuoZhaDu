from app.data.utils import (
    calculate_open_close_time,
    health,
    calculate_pump_run_time,
)


def task1(unit):
    # 开关机时间计算,   统计
    # 反调次数计算,     统计
    # 电磁阀超时计算,   统计
    # 曲线拟合度计算
    calculate_open_close_time(unit)

    # 泵运行时间计算,统计
    calculate_pump_run_time(unit)


def task2(unit):
    # 综合打分健康度
    health(unit)
