import numpy as np
import pandas as pd
import json
from functools import reduce
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Timer, Counter, Alarm
import redis
from parameters import Constant
from concurrent.futures import ThreadPoolExecutor, wait
from scipy import stats, signal


redis_client = redis.StrictRedis(host="127.0.0.1", port=6379, db=0)


def test_mqtt():
    unit = 1
    obj = 'pump2'
    statistics_timer(unit, obj)


def test_mqtt1():
    unit = 1
    obj = 'close'
    statistics_timer(unit, obj)


def test_mqtt2():
    unit = 1
    event = 'close'
    statistics_retro(unit, event)

def test_mqtt3():
    unit = 1
    statistics_overtime(unit)

def test_mqtt4():
    unit = 1
    event = 'close'
    goodness_of_fit(unit, event)

def test_mqtt5():
    unit = 1
    from time import time
    t1 = time()
    with ThreadPoolExecutor(max_workers=Constant.OPEN_CLOSE_POOL_NUM) as pool:
        job1 = pool.submit(statistics_retro, unit, "close")
        job2 = pool.submit(statistics_timer, unit, "close")
        # if unit <= 6:
        #     pool.submit(solenoid_valve_overtime1, unit, close_time1, close_time2)
        # else:
        #     pool.submit(solenoid_valve_overtime2, unit, close_time1, close_time2)
        job3 = pool.submit(statistics_overtime, unit)
        # job4 = pool.submit(goodness_of_fit, unit, "close")
        wait([job1, job2, job3])
    # statistics_retro(unit, "close")
    # statistics_timer(unit, "close")
    # statistics_overtime(unit)
    # goodness_of_fit(unit, "close")
    print(time() - t1)

# 初始化全局变量
cv1_open_time1 = cv1_close_time1 = \
cv2_open_time1 = cv2_close_time1 = \
cv3_open_time1 = cv3_close_time1 = \
cv4_open_time1 = cv4_close_time1 = \
cv5_open_time1 = cv5_close_time1 = \
cv6_open_time1 = cv6_close_time1 = \
cv7_open_time1 = cv7_close_time1 = \
cv8_open_time1 = cv8_close_time1 = \
cv9_open_time1 = cv9_close_time1 = \
cv1_pump1_time1 = cv1_pump2_time1 = cv1_pump3_time1 = \
cv2_pump1_time1 = cv2_pump2_time1 = cv2_pump3_time1 = \
cv3_pump1_time1 = cv3_pump2_time1 = cv3_pump3_time1 = \
cv4_pump1_time1 = cv4_pump2_time1 = cv4_pump3_time1 = \
cv5_pump1_time1 = cv5_pump2_time1 = cv5_pump3_time1 = \
cv6_pump1_time1 = cv6_pump2_time1 = cv6_pump3_time1 = \
cv7_pump1_time1 = cv7_pump2_time1 = cv7_pump3_time1 = \
cv8_pump1_time1 = cv8_pump2_time1 = cv8_pump3_time1 = \
cv9_pump1_time1 = cv9_pump2_time1 = cv9_pump3_time1 = \
pd.to_datetime(datetime.now())

engine = create_engine("mysql+pymysql://root:root@10.64.25.104:3306/dn")

def get_data_from_db(table, columns, term="", return_type="pandas"):
    if not term:
        term = f"where time >= now() - interval {Constant.DEFAULT_INTERVAL} second"

    query_sql = f"select {columns} from {table} {term}"

    df = pd.read_sql_query(sql=query_sql, con=engine)
    # df = cx.read_sql(conn=Config.MYSQL_DATABASE_URI, query=query_sql, return_type=return_type)

    return df


def calculate_retroegulation(unit, begin, end):
    """统计6个接力器反调次数"""
    retro1 = retro2 = retro3 = retro4 = retro5 = retro6 = 0

    table = f"cv{unit}_digital_1s_" + datetime.today().strftime("%Y%m%d")
    columns = "ALM_SMPOS1_ERR, ALM_SMPOS2_ERR, ALM_SMPOS3_ERR, ALM_SMPOS4_ERR, ALM_SMPOS5_ERR, ALM_SMPOS6_ERR"
    term = f"where time >= '{begin}' and time <= '{end}'"

    df = get_data_from_db(table, columns, term)

    # retro1 = np.sum(df["ALM_SMPOS1_ERR"] == "01")
    # retro2 = np.sum(df["ALM_SMPOS2_ERR"] == "01")
    # retro3 = np.sum(df["ALM_SMPOS3_ERR"] == "01")
    # retro4 = np.sum(df["ALM_SMPOS4_ERR"] == "01")
    # retro5 = np.sum(df["ALM_SMPOS5_ERR"] == "01")
    # retro6 = np.sum(df["ALM_SMPOS6_ERR"] == "01")

    for err1, err2, err3, err4, err5, err6 in zip(
        df["ALM_SMPOS1_ERR"],
        df["ALM_SMPOS2_ERR"],
        df["ALM_SMPOS3_ERR"],
        df["ALM_SMPOS4_ERR"],
        df["ALM_SMPOS5_ERR"],
        df["ALM_SMPOS6_ERR"],
    ):
        if err1 == "01":
            retro1 += 1
        elif err2 == "01":
            retro2 += 1
        elif err3 == "01":
            retro3 += 1
        elif err4 == "01":
            retro4 += 1
        elif err5 == "01":
            retro5 += 1
        elif err6 == "01":
            retro6 += 1

    return retro1, retro2, retro3, retro4, retro5, retro6


def calculate_open_close_time(unit):
    """计算球阀开关机时间, 反调次数统计,电磁阀超时判定也放在开关完筒阀后执行"""

    table = f"cv{unit}_digital_0s_" + datetime.today().strftime("%Y%m%d")
    columns = "time, RINGGATE_CLOSED, RINGGATE_OPENNED"

    df = get_data_from_db(table, columns)

    DBSession = sessionmaker(bind=engine)

    for t, closed, opened in zip(
        df["time"], df["RINGGATE_CLOSED"], df["RINGGATE_OPENNED"]
    ):
        if closed == "10":  # 全关 -> 开
            exec(f"cv{unit}_open_time1 = t")
        elif closed == "01":  # 开 -> 全关
            close_time2 = t
            close_time1 = 0
            exec(f"close_time1 = cv{unit}_close_time1")
            close_time = (close_time2 - close_time1).to_pytimedelta()

            if close_time > timedelta(seconds=Constant.MIN_CLOSE_TIME):
                retro = calculate_retroegulation(unit, close_time1, close_time2)
                with DBSession() as session:
                    session.add_all(
                        [
                            Timer(time=t, info=f"cv{unit}_close", val=close_time),
                            Counter(time=t, info=f"cv{unit}_retro1_close", val=retro[0]),
                            Counter(time=t, info=f"cv{unit}_retro2_close", val=retro[1]),
                            Counter(time=t, info=f"cv{unit}_retro3_close", val=retro[2]),
                            Counter(time=t, info=f"cv{unit}_retro4_close", val=retro[3]),
                            Counter(time=t, info=f"cv{unit}_retro5_close", val=retro[4]),
                            Counter(time=t, info=f"cv{unit}_retro6_close", val=retro[5]),
                        ]
                    )
                    session.commit()
                
                with ThreadPoolExecutor(max_workers=Constant.OPEN_CLOSE_POOL_NUM) as pool:
                    job1 = pool.submit(statistics_retro, unit, "close")
                    job2 = pool.submit(statistics_timer, unit, "close")
                    if unit <= 6:
                        job3 = pool.submit(solenoid_valve_overtime1, unit, close_time1, close_time2)
                    else:
                        job3 = pool.submit(solenoid_valve_overtime2, unit, close_time1, close_time2)
                    job4 = pool.submit(statistics_overtime, unit)
                    job5 = pool.submit(goodness_of_fit, unit, "close")

                    wait([job1, job2, job3, job4, job5])
                    
                # statistics_retro(unit, "close")

                # statistics_timer(unit, "close")

                # if unit <= 6:
                #     solenoid_valve_overtime1(unit, close_time1, close_time2)
                # else:
                #     solenoid_valve_overtime2(unit, close_time1, close_time2)
                # statistics_overtime(unit)

                # goodness_of_fit(unit, "close")

        if opened == "10":  # 全开 -> 关
            exec(f"cv{unit}_close_time1 = t")
        elif opened == "01":  # 关 -> 全开
            open_time2 = t
            open_time1 = 0
            exec(f"open_time1 = cv{unit}_open_time1")
            open_time = (open_time2 - open_time1).to_pytimedelta()

            if open_time > timedelta(seconds=Constant.MIN_OPEN_TIME):
                retro = calculate_retroegulation(unit, open_time1, open_time2)
                with DBSession() as session:
                    session.add_all(
                        [
                            Timer(time=t, info=f"cv{unit}_open", val=open_time),
                            Counter(time=t, info=f"cv{unit}_retro1_open", val=retro[0]),
                            Counter(time=t, info=f"cv{unit}_retro2_open", val=retro[1]),
                            Counter(time=t, info=f"cv{unit}_retro3_open", val=retro[2]),
                            Counter(time=t, info=f"cv{unit}_retro4_open", val=retro[3]),
                            Counter(time=t, info=f"cv{unit}_retro5_open", val=retro[4]),
                            Counter(time=t, info=f"cv{unit}_retro6_open", val=retro[5]),
                        ]
                    )
                    session.commit()

                with ThreadPoolExecutor(max_workers=Constant.OPEN_CLOSE_POOL_NUM) as pool:
                    job1 = pool.submit(statistics_retro, unit, "open")
                    job2 = pool.submit(statistics_timer, unit, "open")
                    if unit <= 6:
                        job3 = pool.submit(solenoid_valve_overtime1, unit, open_time1, open_time2)
                    else:
                        job3 = pool.submit(solenoid_valve_overtime2, unit, open_time1, open_time2)
                    job4 = pool.submit(statistics_overtime, unit)
                    job5 = pool.submit(goodness_of_fit, unit, "open")

                    wait([job1, job2, job3, job4, job5])

                # statistics_retro(unit, "open")

                # statistics_timer(unit, "open")

                # if unit <= 6:
                #     solenoid_valve_overtime1(unit, open_time1, open_time2)
                # else:
                #     solenoid_valve_overtime2(unit, open_time1, open_time2)
                # statistics_overtime(unit)

                # goodness_of_fit(unit, "open")


def statistics_timer(unit, obj):
    """计算各统计量
    obj -> str: pump1/2/3, open, close
    topic -> cv{unit}_{obj}: cv1_close/pump1
    """
    table = "timer"
    columns = "*"
    term = f"where info = 'cv{unit}_{obj}' order by time desc limit {Constant.TIMER_DATA_LIMIT}"

    df = get_data_from_db(table, columns, term)
    df["val"] = df["val"].values.astype("timedelta64[s]")

    mean_diff = -np.diff(df["time"]).mean() / np.timedelta64(1, "s")
    max_ = max(df["val"])
    min_ = min(df["val"])
    mean = np.mean(df["val"])
    timestamp = [df["time"][i].strftime("%Y-%m-%d %H:%M:%S.%f") for i in range(len(df))]
    val = df["val"].tolist()

    data = {
        "timestamp": timestamp,
        "value": val,
        "mean_diff": mean_diff,
        "max": max_,
        "min": min_,
        "mean": mean,
    }

    redis_client.hset(f"cv{unit}", obj, json.dumps(data))


def statistics_retro(unit, event):
    """计算接力器反向调节次数的统计量
    event -> str: open, close
    topic -> cv{unit}_retro_{event}: cv1_retro_close
    """
    table = "counter"
    columns = "*"
    term = f"where info like '%%{event}' order by time desc limit {Constant.RETRO_DATA_LIMIT}"

    df = get_data_from_db(table, columns, term)

    df = pd.pivot_table(df, values="val", index="time", columns="info")

    max_ = np.max(df, axis=0).tolist()
    min_ = np.min(df, axis=0).tolist()
    mean = np.mean(df, axis=0).tolist()
    timestamp = [str(df.index[i]) for i in range(len(df))]
    keys = [
        f"cv{unit}_retro1_{event}", 
        f"cv{unit}_retro2_{event}", 
        f"cv{unit}_retro3_{event}", 
        f"cv{unit}_retro4_{event}", 
        f"cv{unit}_retro5_{event}", 
        f"cv{unit}_retro6_{event}", 
    ]
    val = df.values.tolist() # 二维数组需先提取values再转list

    data = {
        "timestamp": timestamp,
        "key": keys,
        "value": val,
        "max": max_,
        "min": min_,
        "mean": mean,
    }

    redis_client.hset(f"cv{unit}", f"retro_{event}", json.dumps(data))


def statistics_overtime(unit):
    """电磁阀超时次数统计
    topic -> cv{unit}_overtime: cv1_overtime
    """
    table = "alarm"
    columns = "info, count(info)"
    term = f"where info like 'cv{unit}%%' and time >= now() - interval {Constant.OVERTIME_DATA_INTERVAL_DAYS} day group by info"

    # pip install pyarrow, it is quicker using count()
    df = get_data_from_db(table, columns, term)
    # df = table.to_pandas(split_blocks=False, date_as_object=False)

    data = {
        "key": df["info"].tolist(),
        "value": df["count(info)"].tolist()
    }

    redis_client.hset(f"cv{unit}", "overtime", json.dumps(data))


def get_relay_displacement(unit, event):
    """获取接力器位移, 用于计算曲线拟合度
    event: 1 -> 开, 0 -> 关
    """
    table = "timer"
    columns = "*"
    term = f"where info = 'cv{unit}_{event}' order by time desc limit 11"

    df = get_data_from_db(table, columns, term)
    df["start_time"] = df["time"] - df["val"]

    table = f"a_cv{unit}_wave_recording"
    columns = "POSITION_SM1, POSITION_SM2, POSITION_SM3, POSITION_SM4, POSITION_SM5, POSITION_SM6"
    dfs = []

    for begin, end in zip(df["start_time"], df["time"]):
        term = f"where type = {event} and time >= '{begin}' and time <= '{end}'"
        df = get_data_from_db(table, columns, term)
        # 取6个接力器的平均值
        df = np.mean(df, axis=1)
        # 降采样,为了得到等长数组
        df = signal.resample(df, Constant.RELAY_DISPLACEMENT_SAMPLING_POINTS)
        # resample使用了fft,所以要去除两头的点
        dfs.append(df[Constant.RELAY_DISPLACEMENT_TRIM_POINTS: -Constant.RELAY_DISPLACEMENT_TRIM_POINTS])

    return dfs


def R2_fun(y_actual, y_ideal):
    """拟合优度R^2 接近1好"""
    return 1 - (np.sum((y_actual - y_ideal) ** 2)) / (
        np.sum((np.mean(y_ideal) - y_ideal) ** 2)
    )


def E_fun(y_ideal, y_actual, power=2, k=1, epsilon=Constant.EPSILON):
    """拟合度E, R方改进版
    引用: https://github.com/jackyrj/myblog/blob/master/%E6%8B%9F%E5%90%88%E5%BA%A6/main.py
    """
    y_mean = np.mean(y_ideal)
    z = (
        (np.sum((np.abs(y_actual - y_ideal)) ** power))
        / (np.sum((abs(y_ideal - y_mean)) ** power) + epsilon)
        * k
    )
    return 1 / (1 + (z) ** (1 / power))


def rmse(y_ideal, y_actual):
    mse = np.sum((y_actual - y_ideal) ** 2) / len(y_ideal)
    return mse ** 0.5


def goodness_of_fit(unit, event):
    dfs = get_relay_displacement(unit, event)

    E_GOF, RMSE, KSTest, WD = [], [], [], []

    with ThreadPoolExecutor(max_workers=Constant.GOF_POOL_NUM) as pool:
        for curve_history in dfs[1:]:
            job1 = pool.submit(E_fun, dfs[0], curve_history)
            job2 = pool.submit(rmse, dfs[0], curve_history)
            # 0假设:来自不同连续性分布
            job3 = pool.submit(stats.ks_2samp, dfs[0], curve_history)
            # 分布间距离
            job4 = pool.submit(stats.wasserstein_distance, dfs[0], curve_history)
            
            E_GOF.append(job1.result())
            RMSE.append(job2.result())
            KSTest.append(job3.result().pvalue) # 大于0.05则拒绝0假设
            WD.append(job4.result())

    # for curve_history in dfs[1:]:
    #     R2.append(R2_fun(dfs[0], curve_history))
    #     E.append(E_fun(dfs[0], curve_history))
    #     RMSE.append(rmse(dfs[0], curve_history))
    #     MAE.append(mae(dfs[0], curve_history))

    data = {
        "timestamps": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "key": ["拟合度E", "均方根误差", "KS检验", "Wasserstein距离"],
        "value": [E_GOF, RMSE, KSTest, WD]
    }

    redis_client.hset(f"cv{unit}", f"fit_{event}", json.dumps(data, ensure_ascii=False))


def solenoid_valve_overtime1(unit, begin, end):
    """判断1-6号机电磁阀动作超时(超过1周期)"""
    table = f"cv{unit}_digital_0s_" + datetime.today().strftime("%Y%m%d")

    columns = """time, VAL_102_POS_A, VAL_102_POS_B, VAL_103_POS_A, VAL_103_POS_B, \
VAL_105_POS_A, VAL_105_POS_B, VAL_108_POS_A, VAL_108_POS_A, \
VAL_102_A_CTRLOUT, VAL_102_B_CTRLOUT, VAL_103_A_CTRLOUT, VAL_103_B_CTRLOUT, \
VAL_105_A_CTRLOUT, VAL_105_B_CTRLOUT, VAL_108_A_CTRLOUT, VAL_108_B_CTRLOUT\
"""

    term = f"where time >= '{begin - timedelta(seconds=Constant.ACTION_TIME_SV_BEFORE_CV)}' and time <= '{end}'"

    df = get_data_from_db(table, columns, term)

    DBSession = sessionmaker(bind=engine)

    with DBSession() as session:
        for (
            t,
            posA102,
            posB102,
            posA103,
            posB103,
            posA105,
            posB105,
            posA108,
            posB108,
            ctrlA102,
            ctrlB102,
            ctrlA103,
            ctrlB103,
            ctrlA105,
            ctrlB105,
            ctrlA108,
            ctrlB108,
        ) in zip(
            df["time"],
            df["VAL_102_POS_A"],
            df["VAL_102_POS_B"],
            df["VAL_103_POS_A"],
            df["VAL_103_POS_B"],
            df["VAL_105_POS_A"],
            df["VAL_105_POS_B"],
            df["VAL_108_POS_A"],
            df["VAL_108_POS_B"],
            df["VAL_102_A_CTRLOUT"],
            df["VAL_102_B_CTRLOUT"],
            df["VAL_103_A_CTRLOUT"],
            df["VAL_103_B_CTRLOUT"],
            df["VAL_105_A_CTRLOUT"],
            df["VAL_105_B_CTRLOUT"],
            df["VAL_108_A_CTRLOUT"],
            df["VAL_108_B_CTRLOUT"],
        ):
            if ctrlA102[0] == "1" and posA102 == "00":
                session.add(Alarm(time=t, info=f"cv{unit}_sv102a_overtime", val=1))
            if ctrlB102[0] == "1" and posB102 == "00":
                session.add(Alarm(time=t, info=f"cv{unit}_sv102b_overtime", val=1))
            if ctrlA103[0] == "1" and posA103 == "00":
                session.add(Alarm(time=t, info=f"cv{unit}_sv103a_overtime", val=1))
            if ctrlB103[0] == "1" and posB103 == "00":
                session.add(Alarm(time=t, info=f"cv{unit}_sv103b_overtime", val=1))
            if ctrlA105[0] == "1" and posA105 == "00":
                session.add(Alarm(time=t, info=f"cv{unit}_sv105a_overtime", val=1))
            if ctrlB105[0] == "1" and posB105 == "00":
                session.add(Alarm(time=t, info=f"cv{unit}_sv105b_overtime", val=1))
            if ctrlA108[0] == "1" and posA108 == "00":
                session.add(Alarm(time=t, info=f"cv{unit}_sv108a_overtime", val=1))
            if ctrlB108[0] == "1" and posB108 == "00":
                session.add(Alarm(time=t, info=f"cv{unit}_sv108b_overtime", val=1))

        session.commit()

def solenoid_valve_overtime2(unit, begin, end):
    """判断7-9号机电磁阀动作超时(超过1周期)"""
    table = f"cv{unit}_digital_0s_" + datetime.today().strftime("%Y%m%d")
    columns = """time, VAL_AA30_A_CTRLOUT, VAL_AA30_B_CTRLOUT, VAL_AA40_CTRLOUT, \
VAL_AA30_POS_A, VAL_AA30_POS_B, VAL_AA40_POS
"""
    term = f"where time >= '{begin - timedelta(seconds=Constant.ACTION_TIME_SV_BEFORE_CV)}' and time <= '{end}'"

    df = get_data_from_db(table, columns, term)

    DBSession = sessionmaker(bind=engine)

    with DBSession() as session:
        for (
            t,
            posA30,
            posB30,
            pos40,
            ctrlA30,
            ctrlB30,
            ctrl40
        ) in zip(
            df["time"],
            df["VAL_AA30_A_CTRLOUT"],
            df["VAL_AA30_B_CTRLOUT"],
            df["VAL_AA40_CTRLOUT"],
            df["VAL_AA30_POS_A"],
            df["VAL_AA30_POS_B"],
            df["VAL_AA40_POS"]
        ):
            if ctrlA30[0] == "1" and posA30 == "00":
                session.add(Alarm(time=t, info=f"cv{unit}_sv30a_overtime", val=1))
            if ctrlB30[0] == "1" and posB30 == "00":
                session.add(Alarm(time=t, info=f"cv{unit}_sv30b_overtime", val=1))
            if ctrl40[0] == "1" and pos40 == "00":
                session.add(Alarm(time=t, info=f"cv{unit}_sv40_overtime", val=1))

        session.commit()


def calculate_pump_run_time(unit):
    """计算3个泵运行时间"""
    pump_time = timedelta(0)

    table = f"cv{unit}_digital_0s_" + datetime.today().strftime("%Y%m%d")
    columns = "time, PUMP1_CTRLOUT, PUMP2_CTRLOUT, PUMP_STANDBY_RUN"

    df = get_data_from_db(table, columns)

    DBSession = sessionmaker(bind=engine)

    for t, pump1, pump2, pump3 in zip(
        df["time"], df["PUMP1_CTRLOUT"], df["PUMP2_CTRLOUT"], df["PUMP_STANDBY_RUN"]
    ):
        if pump1 == "01":  # 停 -> 启
            exec(f"cv{unit}_pump1_time1 = t")
        elif pump1 == "10":  # 启 -> 停
            exec(f"pump_time = (t - cv{unit}_pump1_time1).to_pytimedelta()")

            if pump_time > timedelta(seconds=Constant.MIN_PUMP_TIME):
                with DBSession() as session:
                    session.add(
                        Timer(time=t, info=f"cv{unit}_pump1", val=pump_time),
                    )
                session.commit()

                statistics_timer(unit, "pump1")

        if pump2 == "01":  # 停 -> 启
            exec(f"cv{unit}_pump2_time1 = t")
        elif pump2 == "10":  # 启 -> 停
            exec(f"pump_time = (t - cv{unit}_pump2_time1).to_pytimedelta()")

            if pump_time > timedelta(seconds=Constant.MIN_PUMP_TIME):
                with DBSession() as session:
                    session.add(
                        Timer(time=t, info=f"cv{unit}_pump2", val=pump_time),
                    )
                session.commit()

                statistics_timer(unit, "pump2")

        if pump3 == "01":  # 停 -> 启
            exec(f"cv{unit}_pump3_time1 = t")
        elif pump3 == "10":  # 启 -> 停
            exec(f"(pump_time = t - cv{unit}_pump3_time1).to_pytimedelta()")

            if pump_time > timedelta(seconds=Constant.MIN_PUMP_TIME):
                with DBSession() as session:
                    session.add(
                        Timer(time=t, info=f"cv{unit}_pump3", val=pump_time),
                    )
                session.commit()

                statistics_timer(unit, "pump3")


def get_one_health_df(table, term):
    hz = (
        Constant.HEALTH_MS_DATA_SAMPLING_HZ
        if "0s" in table
        else Constant.HEALTH_S_DATA_SAMPLING_HZ
    )

    # pd.read_sql: use %% as % for python3.9
    term += f" and rownum %% {hz} = 1"
    columns = "*"

    query_sql = (
        f"select {columns} from (select {columns}, @row := @row + 1 as rownum "
        + f"from (select @row := 0) r, {table}) ranked {term}"
    )

    df = pd.read_sql_query(sql=query_sql, con=engine)
    # df = cx.read_sql(conn=Config.MYSQL_DATABASE_URI, query=query_sql)

    # "2:-1" remove the column @row := 0, time, rownum
    return df.iloc[:, 2:-1]


def get_health_df(unit):
    now = datetime.now()
    today = now.strftime("%Y%m%d")
    yesterday = (now - timedelta(days=Constant.HEALTH_DATA_INTERVAL_DAYS)).strftime("%Y%m%d")
    Constant.HEALTH_DATA_INTERVAL_DAYS = 30 # for test
    today = yesterday = '20221103' # for test

    table1 = f"cv{unit}_digital_0s_" + today
    table2 = f"cv{unit}_digital_1s_" + today
    table3 = f"cv{unit}_analog_0s_" + today
    table4 = f"cv{unit}_analog_1s_" + today

    table5 = f"cv{unit}_digital_0s_" + yesterday
    table6 = f"cv{unit}_digital_1s_" + yesterday
    table7 = f"cv{unit}_analog_0s_" + yesterday
    table8 = f"cv{unit}_analog_1s_" + yesterday

    # dfs = [get_one_health_df(table) for table in tables]

    # today_df = reduce(lambda left, right: left.join(right), dfs[:4])
    # yesterday_df = reduce(lambda left, right: left.join(right), dfs[4:])

    term_today = f"where time <= now()"
    term_yesterday = f"where time >= now() - interval {Constant.HEALTH_DATA_INTERVAL_DAYS} day"

    with ThreadPoolExecutor(max_workers=Constant.GET_HEALTH_DF_POOL_NUM) as pool:
        job1 = pool.submit(get_one_health_df, table1, term_today)
        job2 = pool.submit(get_one_health_df, table2, term_today)
        job3 = pool.submit(get_one_health_df, table3, term_today)
        job4 = pool.submit(get_one_health_df, table4, term_today)
        job5 = pool.submit(get_one_health_df, table5, term_yesterday)
        job6 = pool.submit(get_one_health_df, table6, term_yesterday)
        job7 = pool.submit(get_one_health_df, table7, term_yesterday)
        job8 = pool.submit(get_one_health_df, table8, term_yesterday)

    today_df = reduce(
        lambda left, right: left.join(right), 
        [job1.result(),job2.result(),job3.result(),job4.result()]
    )
    yesterday_df = reduce(
        lambda left, right: left.join(right), 
        [job5.result(),job6.result(),job7.result(),job8.result()]
    )

    final_df = pd.concat([yesterday_df, today_df])

    final_df.fillna(method="ffill", inplace=True)

    for col in final_df.columns:
        if final_df[col].dtype == object:
            final_df[col] = final_df[col].values.astype("int8")

    return final_df


def zscore_test(df):
    """标准分数
    与均值超过3个标准差则认为是异常点
    每列异常点多于3%则认为改列异常,返回异常列的比例
    """
    zs = stats.zscore(df, axis=1, nan_policy="omit").values

    return np.sum(np.sum(np.abs(zs) > 3, axis=1) / zs.shape[0] > 0.03) / zs.shape[1]


def oneway_anova_test(df):
    """单因素方差分析,组间两两差异
    检测半天时间的组间均值差异,pvalue>0.05则接收零假设,差异不明显
    和turkey test类似,turkey test可以知道两个分组对整体差异的贡献度
    返回差异列的比例
    """
    half = df.shape[0] // 2

    p_value = stats.f_oneway(df.iloc[:half, :], df.iloc[half:, :]).pvalue

    return np.sum(p_value < 0.05) / df.shape[1]


def grubbs_test(df):
    """格拉布斯法, 检测出离群数据
    测试需要总体是正态分布
    给出每列离群数据的索引值,没有则为[]
    返回有异常值的列的比例
    """
    from outliers import smirnov_grubbs as grubbs

    x = 0
    for i in range(df.shape[1]):
        x += len(grubbs.two_sided_test_indices(df.iloc[:, i].values, alpha=0.05)) > 0

    return x / df.shape[1]


def sos(df):
    """随机异常值选择, 基于矩阵的关联度, 耗时较长
    返回异常列的比例
    """
    from pyod.models import sos

    clf = sos.SOS(contamination=0.1)
    result = clf.fit_predict(df)

    return np.sum(result) / df.shape[1]


def ecod(df):
    """Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions
    经验累计分布函数的无监督离群值检测"""
    from pyod.models import ecod

    clf = ecod.ECOD(contamination=0.1)
    result = clf.fit_predict(df)

    return np.sum(result) / df.shape[1]


def elliptic_envelope(df):
    """椭圆模型拟合 假设数据服从高斯分布并学习一个椭圆 服务器上耗时长"""
    from sklearn.covariance import EllipticEnvelope

    clf = EllipticEnvelope(contamination=0.1)
    result = clf.fit_predict(df)

    return np.sum(result == -1) / df.shape[1]


def lscp(df):
    """LSCP: Locally Selective Combination in Parallel Outlier Ensembles 多个异常检测算法的并行集成框架
    LOF: Local outlier factor 局部异常因素, 基于密度
    COF: Connectivity-Based Outlier Factor 基于连通性的异常因素
    INNE: Isolation-based Anomaly Detection Using Nearest-Neighbor Ensembles 孤立森林升级版
    PCA: 主成分分析 降维
    OCSVM: 单类支持向量机
    返回异常列的比例
    """
    from pyod.models.lscp import LSCP
    from pyod.models import lof, cof, inne, pca, ocsvm

    detector_list = [
        lof.LOF(),
        cof.COF(),
        inne.INNE(),
        pca.PCA(),
        ocsvm.OCSVM(),
    ]
    clf = LSCP(contamination=0.1, detector_list=detector_list)
    result = clf.fit_predict(df)
    return np.sum(result == 1) / df.shape[1]


def spectral_residual_saliency(df):
    """ 残差谱显著性分析,异常分数>99分数
    返回异常列的比例
    """
    import sranodec as anom

    spec = anom.Silency(amp_window_size=24, series_window_size=24, score_window_size=100)

    abnormal = 0
    for i in range(df.shape[1]):
        score = spec.generate_anomaly_score(df.values[:, i])
        abnormal += np.sum(score > np.percentile(score, 99)) > 0

    return abnormal / df.shape[1]

def health(unit):
    from time import time
    time1 = time()
    # df = get_health_df(unit) for test
    df = pd.read_csv("final_df.csv")
    if len(df) == 0:
        return

    df = df + Constant.EPSILON

    with ThreadPoolExecutor(max_workers=Constant.HEALTH_POO_NUM) as pool:
        job1 = pool.submit(zscore_test, df)
        job2 = pool.submit(oneway_anova_test, df)
        job3 = pool.submit(grubbs_test, df)
        job4 = pool.submit(ecod, df)
        # job5 = pool.submit(elliptic_envelope, df)
        job5 = pool.submit(lscp, df)
        job6 = pool.submit(spectral_residual_saliency, df)

        zscore = job1.result()
        f_oneway = job2.result()
        grubbs = job3.result()
        ecod_result = job4.result()
        # ee = job5.result()
        lscp_result = job5.result()
        sr = job6.result()

    # zscore = zscore_test(df)
    # f_oneway = oneway_anova_test(df)
    # grubbs = grubbs_test(df)
    # ecod_result = ecod(df)
    # ee = elliptic_envelope(df)
    # lscp_result = lscp(df)
    # sr = spectral_residual_saliency(df)


    data = {
        "timestamps": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "key": ["标准分数", "单因素方差分析", "Grubbs检验", "ECOD检测", "LSCP检测", "残差谱分析"],
        "value": [zscore, f_oneway, grubbs, ecod_result, lscp_result, sr]
    }

    redis_client.hset(f"cv{unit}", "health", json.dumps(data, ensure_ascii=False))

    print(time() - time1)