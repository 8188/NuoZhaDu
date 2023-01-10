#ifdef __unix__         
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
#elif defined(_WIN32) || defined(WIN32) 
# distutils: extra_compile_args = /openmp
# distutils: extra_link_args = /openmp
#endif
# cython: boundscheck=False

import numpy as np
cimport numpy as np
import pandas as pd
import json
from functools import reduce
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import Config
from app.data.models import Timer, Counter, Alarm
from app import redis_client
from app.data.parameters import Constant
from app import logger
cimport cython
from cython.parallel cimport prange
from concurrent.futures import ThreadPoolExecutor, wait
from scipy import stats, signal


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

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)

cdef get_data_from_db(str table, str columns, str term=""):
    if not term:
        term = f"where time >= now() - interval {Constant.cdefAULT_INTERVAL} second"

    cdef str query_sql = f"select {columns} from {table} {term}"

    df = pd.read_sql_query(sql=query_sql, con=engine)

    return df


cdef tuple calculate_retroegulation(int unit, begin, end):
    """统计6个接力器反调次数"""
    cdef int retro1 = 0
    cdef int retro2 = 0
    cdef int retro3 = 0
    cdef int retro4 = 0
    cdef int retro5 = 0
    cdef int retro6 = 0

    cdef str table = f"cv{unit}_digital_1s_" + datetime.today().strftime("%Y%m%d")
    cdef str columns = "ALM_SMPOS1_ERR, ALM_SMPOS2_ERR, ALM_SMPOS3_ERR, ALM_SMPOS4_ERR, ALM_SMPOS5_ERR, ALM_SMPOS6_ERR"
    cdef str term = f"where time >= '{begin}' and time <= '{end}'"

    df = get_data_from_db(table, columns, term)

    cdef:
        int[:,:] array = df.values.astype("int32")
        ssize_t length = len(df)
        int i = 0

    for i in prange(length, nogil=True):
        if array[i][0] == 1:
            retro1 += 1
        if array[i][1] == 1:
            retro2 += 1
        if array[i][2] == 1:
            retro3 += 1
        if array[i][3] == 1:
            retro4 += 1
        if array[i][4] == 1:
            retro5 += 1
        if array[i][5] == 1:
            retro6 += 1

    return retro1, retro2, retro3, retro4, retro5, retro6


cpdef calculate_open_close_time(int unit):
    """计算球阀开关机时间, 反调次数统计,电磁阀超时判定也放在开关完筒阀后执行"""

    cdef str table = f"cv{unit}_digital_0s_" + datetime.today().strftime("%Y%m%d")
    cdef str columns = "time, RINGGATE_CLOSED, RINGGATE_OPENNED"

    df = get_data_from_db(table, columns)

    DBSession = sessionmaker(bind=engine)

    cdef str closed = ""
    cdef str opened = ""
    cdef tuple retro = ()

    for t, closed, opened in zip(
        df["time"], df["RINGGATE_CLOSED"], df["RINGGATE_OPENNED"]
    ):
        if closed == "10":  # 全关 -> 开
            exec(f"cv{unit}_open_time1 = t")
        elif closed == "01":  # 开 -> 全关
            close_time2 = t
            close_time1 = 0
            exec(f"close_time1 = cv{unit}_close_time1")
            close_time = (close_time2 - close_time1) / np.timedelta64(1, "s")

            if close_time > timedelta(seconds=Constant.MIN_CLOSE_TIME):
                logger.info(f"CV{unit} Closed")
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

                    wait(job1, job2, job3, job4, job5)


        if opened == "10":  # 全开 -> 关
            exec(f"cv{unit}_close_time1 = t")
        elif opened == "01":  # 关 -> 全开
            open_time2 = t
            open_time1 = 0
            exec(f"open_time1 = cv{unit}_open_time1")
            open_time = (open_time2 - open_time1) / np.timedelta64(1, "s")

            if open_time > timedelta(seconds=Constant.MIN_OPEN_TIME):
                logger.info(f"CV{unit} Opened")
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

                    wait(job1, job2, job3, job4, job5)


# 用cpdef,否则多线程重定义,bug?
cpdef statistics_timer(int unit, str obj):
    """计算各统计量
    obj -> str: pump1/2/3, open, close
    topic -> cv{unit}_{obj}: cv1_close/pump1
    """
    cdef str table = "timer"
    cdef str columns = "*"
    cdef str term = f"where info = 'cv{unit}_{obj}' order by time desc limit {Constant.TIMER_DATA_LIMIT}"

    df = get_data_from_db(table, columns, term)

    cdef double mean_diff = -np.diff(df["time"]).mean() / np.timedelta64(1, "s")
    cdef int max_ = max(df["val"])
    cdef int min_ = min(df["val"])
    cdef double mean = np.mean(df["val"])
    cdef list timestamp = [df["time"][i].strftime("%Y-%m-%d %H:%M:%S.%f") for i in range(len(df))]
    cdef list val = df["val"].tolist()

    cdef dict data = {
        "timestamp": timestamp,
        "value": val,
        "mean_diff": mean_diff,
        "max": max_,
        "min": min_,
        "mean": mean,
    }

    redis_client.hset(f"cv{unit}", obj, json.dumps(data))


cdef statistics_retro(int unit, str event):
    """计算接力器反向调节次数的统计量
    event -> str: open, close
    topic -> cv{unit}_retro_{event}: cv1_retro_close
    """
    cdef str table = "counter"
    cdef str columns = "*"
    cdef str term = f"where info like '%%{event}' order by time desc limit {Constant.RETRO_DATA_LIMIT}"

    df = get_data_from_db(table, columns, term)

    df = pd.pivot_table(df, values="val", index="time", columns="info")

    cdef list max_ = np.max(df, axis=0).tolist()
    cdef list min_ = np.min(df, axis=0).tolist()
    cdef list mean = np.mean(df, axis=0).tolist()
    cdef list timestamp = [str(df.index[i]) for i in range(len(df))]
    cdef list val = df.values.tolist() # 二维数组需先提取values再转list
    cdef list keys = [
        f"cv{unit}_retro1_{event}", 
        f"cv{unit}_retro2_{event}", 
        f"cv{unit}_retro3_{event}", 
        f"cv{unit}_retro4_{event}", 
        f"cv{unit}_retro5_{event}", 
        f"cv{unit}_retro6_{event}", 
    ]

    cdef dict data = {
        "timestamp": timestamp,
        "key": keys,
        "value": val,
        "max": max_,
        "min": min_,
        "mean": mean,
    }
    
    redis_client.hset(f"cv{unit}", f"retro_{event}", json.dumps(data))


cdef statistics_overtime(int unit):
    """电磁阀超时次数统计
    topic -> cv{unit}_overtime: cv1_overtime
    """
    cdef str table = "alarm"
    cdef str columns = "info, count(info)"
    cdef str term = f"where info like 'cv{unit}%%' and time >= now() - interval {Constant.OVERTIME_DATA_INTERVAL_DAYS} day group by info"

    # pip install pyarrow, it is quicker using count()
    df = get_data_from_db(table, columns, term)
    # df = table.to_pandas(split_blocks=False, date_as_object=False)

    data = {
        "key": df["info"].tolist(),
        "value": df["count(info)"].tolist()
    }

    redis_client.hset(f"cv{unit}", "overtime", json.dumps(data))


cdef list get_relay_displacement(int unit, str event):
    """获取接力器位移, 用于计算曲线拟合度
    event: 1 -> 开, 0 -> 关
    """
    cdef str table = "timer"
    cdef str columns = "*"
    cdef str term = f"where info = 'cv{unit}_{event}' order by time desc limit {Constant.RELAY_DISPLACEMENT_LIMIT}"

    df = get_data_from_db(table, columns, term)
    df["start_time"] = df["time"] - pd.to_timedelta(df["val"], unit="s")

    table = f"a_cv{unit}_wave_recording"
    columns = "POSITION_SM1, POSITION_SM2, POSITION_SM3, POSITION_SM4, POSITION_SM5, POSITION_SM6"
    cdef list dfs = []
    cdef int event_code = 1 if event == "open" else 0

    for begin, end in zip(df["start_time"], df["time"]):
        term = f"where type = {event_code} and time >= '{begin}' and time <= '{end}'"
        df = get_data_from_db(table, columns, term)
        # 取6个接力器的平均值
        df = np.mean(df, axis=1)
        # 降采样,为了得到等长数组
        df = signal.resample(df, Constant.RELAY_DISPLACEMENT_SAMPLING_POINTS)
        # resample使用了fft,所以要去除两头的点
        dfs.append(df[Constant.RELAY_DISPLACEMENT_TRIM_POINTS: -Constant.RELAY_DISPLACEMENT_TRIM_POINTS])

    return dfs


cdef double R2_fun(y_actual, y_ideal) except ? -1:
    """拟合优度R^2 接近1好"""
    return 1 - (np.sum((y_actual - y_ideal) ** 2)) / (
        np.sum((np.mean(y_ideal) - y_ideal) ** 2)
    )


# 用cpdef,否则多线程重定义,bug?
cpdef double E_fun(
    np.ndarray[np.double_t, ndim=1] y_ideal, 
    np.ndarray[np.double_t, ndim=1] y_actual, 
    int power=2, 
    int k=1, 
    double epsilon=Constant.EPSILON
) except ? -1:
    """拟合度E, R方改进版
    引用: https://github.com/jackyrj/myblog/blob/master/%E6%8B%9F%E5%90%88%E5%BA%A6/main.py
    """
    cdef double y_mean = np.mean(y_ideal)
    cdef double z = (
        (np.sum((np.abs(y_actual - y_ideal)) ** power))
        / (np.sum((abs(y_ideal - y_mean)) ** power) + epsilon)
        * k
    )
    return 1 / (1 + (z) ** (1 / power))


# 用cpdef,否则多线程重定义,bug?
cpdef double rmse(
    np.ndarray[np.double_t, ndim=1] y_ideal, 
    np.ndarray[np.double_t, ndim=1] y_actual
) except ? -1:
    cdef double mse = np.sum((y_actual - y_ideal) ** 2) / len(y_ideal)
    return mse ** 0.5


cdef goodness_of_fit(int unit, str event):
    """ 拟合度量
    topic -> cv{unit}_fit_{event}: cv1_fit_close
    """
    cdef list dfs = get_relay_displacement(unit, event)

    cdef list E_GOF = []
    cdef list RMSE = []
    cdef list KSTest = []
    cdef list WD = []

    cdef np.ndarray[np.double_t, ndim=1] curve_history
    cdef np.ndarray[np.double_t, ndim=1] curve_now = dfs[0]
    
    with ThreadPoolExecutor(max_workers=Constant.GOF_POOL_NUM) as pool:
        for curve_history in dfs[1:]:
            job1 = pool.submit(E_fun, curve_now, curve_history)
            job2 = pool.submit(rmse, curve_now, curve_history)
            # 0假设:来自不同连续性分布
            job3 = pool.submit(stats.ks_2samp, curve_now, curve_history)
            # 分布间距离
            job4 = pool.submit(stats.wasserstein_distance, curve_now, curve_history)

            # E方越接近1越好, 取倒数统一成越小越好，方便前端展示
            E_GOF.append(1 / job1.result())
            RMSE.append(job2.result())
            KSTest.append(1 / job3.result().pvalue) # 大于0.05则拒绝0假设
            WD.append(job4.result())

    # for curve_history in dfs[1:]:
    #     E_GOF.append(E_fun(dfs[0], curve_history))
    #     RMSE.append(rmse(dfs[0], curve_history))
    #     KSTest.append(stats.ks_2samp(dfs[0], curve_history).pvalue)  # 大于0.05则拒绝0假设
    #     WD.append(stats.wasserstein_distance(dfs[0], curve_history))

    cdef dict data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "key": ["1/拟合度E", "均方根误差", "1/KS检验", "Wasserstein距离"],
        "value": [E_GOF, RMSE, KSTest, WD]
    }

    redis_client.hset(f"cv{unit}", f"fit_{event}", json.dumps(data, ensure_ascii=False))


cdef solenoid_valve_overtime1(int unit, begin, end):
    """判断1-6号机电磁阀动作超时(超过1周期)"""
    cdef str table = f"cv{unit}_digital_0s_" + datetime.today().strftime("%Y%m%d")

    cdef str columns = """time, VAL_102_POS_A, VAL_102_POS_B, VAL_103_POS_A, VAL_103_POS_B, \
VAL_105_POS_A, VAL_105_POS_B, VAL_108_POS_A, VAL_108_POS_A, \
VAL_102_A_CTRLOUT, VAL_102_B_CTRLOUT, VAL_103_A_CTRLOUT, VAL_103_B_CTRLOUT, \
VAL_105_A_CTRLOUT, VAL_105_B_CTRLOUT, VAL_108_A_CTRLOUT, VAL_108_B_CTRLOUT\
"""

    cdef str term = f"where time >= '{begin - timedelta(seconds=Constant.ACTION_TIME_SV_BEFORE_CV)}' and time <= '{end}'"

    df = get_data_from_db(table, columns, term)

    DBSession = sessionmaker(bind=engine)

    cdef str posA102 = ""
    cdef str posB102 = ""
    cdef str posA103 = ""
    cdef str posB103 = ""
    cdef str posA105 = ""
    cdef str posB105 = ""
    cdef str posA108 = ""
    cdef str posB108 = ""
    cdef str ctrlA102 = ""
    cdef str ctrlB102 = ""
    cdef str ctrlA103 = ""
    cdef str ctrlB103 = ""
    cdef str ctrlA105 = ""
    cdef str ctrlB105 = ""
    cdef str ctrlA108 = ""
    cdef str ctrlB108 = ""

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


cdef solenoid_valve_overtime2(unit, begin, end):
    """判断7-9号机电磁阀动作超时(超过1周期)"""
    cdef str table = f"cv{unit}_digital_0s_" + datetime.today().strftime("%Y%m%d")
    cdef str columns = """time, VAL_AA30_A_CTRLOUT, VAL_AA30_B_CTRLOUT, VAL_AA40_CTRLOUT, \
VAL_AA30_POS_A, VAL_AA30_POS_B, VAL_AA40_POS
"""
    cdef str term = f"where time >= '{begin - timedelta(seconds=Constant.ACTION_TIME_SV_BEFORE_CV)}' and time <= '{end}'"

    df = get_data_from_db(table, columns, term)

    DBSession = sessionmaker(bind=engine)

    cdef str posA30 = ""
    cdef str posB30 = ""
    cdef str pos40 = ""
    cdef str ctrlA30 = ""
    cdef str ctrlB30 = ""
    cdef str ctrl40 = ""

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


cpdef calculate_pump_run_time(int unit):
    """计算2个泵运行时间"""
    pump_time = timedelta(0)

    cdef str table = f"cv{unit}_digital_0s_" + datetime.today().strftime("%Y%m%d")
    cdef str columns = "time, PUMP1_CTRLOUT, PUMP2_CTRLOUT"

    df = get_data_from_db(table, columns)

    DBSession = sessionmaker(bind=engine)

    cdef str pump1
    cdef str pump2

    for t, pump1, pump2 in zip(
        df["time"], df["PUMP1_CTRLOUT"], df["PUMP2_CTRLOUT"]
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


cdef get_one_health_df(str table, str term):
        cdef int hz = (
            Constant.HEALTH_MS_DATA_SAMPLING_HZ
            if "0s" in table
            else Constant.HEALTH_S_DATA_SAMPLING_HZ
        )

        # pd.read_sql: use %% as % for python3.9
        term += f" and rownum %% {hz} = 1"
        cdef str columns = "*"

        cdef str query_sql = (
            f"select {columns} from (select {columns}, @row := @row + 1 as rownum "
            + f"from (select @row := 0) r, {table}) ranked {term}"
        )

        df = pd.read_sql_query(sql=query_sql, con=engine)

        # "2:-1" remove the column @row := 0, time, rownum
        return df.iloc[:, 2:-1]


cdef get_health_df(int unit):
    now = datetime.now()
    cdef str today = now.strftime("%Y%m%d")
    cdef str yesterday = (now - timedelta(days=Constant.HEALTH_DATA_INTERVAL_DAYS)).strftime("%Y%m%d")

    cdef str table1 = f"cv{unit}_digital_0s_" + today
    cdef str table2 = f"cv{unit}_digital_1s_" + today
    cdef str table3 = f"cv{unit}_analog_0s_" + today
    cdef str table4 = f"cv{unit}_analog_1s_" + today

    cdef str table5 = f"cv{unit}_digital_0s_" + yesterday
    cdef str table6 = f"cv{unit}_digital_1s_" + yesterday
    cdef str table7 = f"cv{unit}_analog_0s_" + yesterday
    cdef str table8 = f"cv{unit}_analog_1s_" + yesterday

    cdef str term_today = f"where time <= now()"
    cdef str term_yesterday = f"where time >= now() - interval {Constant.HEALTH_DATA_INTERVAL_DAYS} day"
    
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

    cdef str col

    for col in final_df.columns:
        if final_df[col].dtype == object:
            final_df[col] = final_df[col].values.astype("int8")

    return final_df


cdef double zscore_test(df) except ? -1:
    """标准分数
    与均值超过3个标准差则认为是异常点
    每列异常点多于3%则认为该列异常,返回异常列的比例
    """
    cdef np.ndarray[np.double_t, ndim=2] zs = stats.zscore(df, axis=1, nan_policy="omit").values

    return np.count_nonzero(np.count_nonzero(np.abs(zs) > 3, axis=1) / zs.shape[0] > 0.03) / zs.shape[1]


cdef double oneway_anova_test(df) except ? -1:
    """单因素方差分析,组间两两差异
    检测半天时间的组间均值差异,pvalue>0.05则接受零假设,差异不明显
    和turkey test类似,turkey test可以知道两个分组对整体差异的贡献度
    返回差异列的比例
    """
    cdef int half = df.shape[0] // 2

    cdef np.ndarray[np.double_t, ndim=1] p_value = stats.f_oneway(df.iloc[:half, :], df.iloc[half:, :]).pvalue

    return np.count_nonzero(p_value < 0.05) / df.shape[1]


cdef double grubbs_test(df) except ? -1:
    """格拉布斯法, 检测出离群数据
    测试需要总体是正态分布
    给出每列离群数据的索引值,没有则为[]
    返回有异常值的列的比例
    """
    from outliers import smirnov_grubbs as grubbs

    cdef double x = 0
    cdef int i

    for i in range(df.shape[1]):
        x += len(grubbs.two_sided_test_indices(df.iloc[:, i].values, alpha=0.05)) > 0

    return x / df.shape[1]


cdef double sos(df) except ? -1:
    """随机异常值选择, 基于矩阵的关联度, 耗时较长
    返回异常列的比例
    """
    from pyod.models import sos

    clf = sos.SOS(contamination=0.1)
    cdef int[::1] result = clf.fit_predict(df)

    return np.count_nonzero(result) / df.shape[1]


cdef double ecod(df) except ? -1:
    """Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions
    经验累计分布函数的无监督离群值检测"""
    from pyod.models import ecod

    clf = ecod.ECOD(contamination=0.1)
    cdef long[::1] result = clf.fit_predict(df)

    return np.count_nonzero(result) / df.shape[1]


cdef double elliptic_envelope(df) except ? -1:
    """椭圆模型拟合 假设数据服从高斯分布并学习一个椭圆 服务器上耗时长"""
    from sklearn.covariance import EllipticEnvelope

    clf = EllipticEnvelope(contamination=0.1)
    cdef np.ndarray[np.int_t, ndim=1] result = clf.fit_predict(df)

    return np.count_nonzero(result == -1) / df.shape[1]


cdef double lscp(df) except ? -1:
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

    cdef list detector_list = [
        lof.LOF(),
        cof.COF(),
        inne.INNE(),
        pca.PCA(),
        ocsvm.OCSVM(),
    ]
    clf = LSCP(contamination=0.1, detector_list=detector_list)
    cdef np.ndarray[np.int_t, ndim=1] result = clf.fit_predict(df)
    return np.count_nonzero(result == 1) / df.shape[1]


cdef double spectral_residual_saliency(df) except ? -1:
    """ 残差谱显著性分析,异常分数>99分位数
    返回异常列的比例
    """
    import sranodec as anom

    spec = anom.Silency(amp_window_size=24, series_window_size=24, score_window_size=100)

    cdef int abnormal = 0
    for i in range(df.shape[1]):
        score = spec.generate_anomaly_score(df.values[:, i])
        abnormal += np.count_nonzero(score > np.percentile(score, 99)) > 0

    return abnormal / df.shape[1]


cpdef health(int unit):
    from time import time
    time1 = time()
    df = get_health_df(unit)
    if len(df) == 0:
        return

    df = df + Constant.EPSILON

    cdef:
        double zscore = 0.0
        double f_oneway = 0.0
        double grubbs = 0.0
        double ecod_result = 0.0
        double lscp_result = 0.0
        double sr = 0.0

    with ThreadPoolExecutor(max_workers=Constant.HEALTH_POO_NUM) as pool:
        job1 = pool.submit(zscore_test, df)
        job2 = pool.submit(oneway_anova_test, df)
        job3 = pool.submit(grubbs_test, df)
        job4 = pool.submit(ecod, df)
        job5 = pool.submit(lscp, df)
        job6 = pool.submit(spectral_residual_saliency, df)

        zscore = job1.result()
        f_oneway = job2.result()
        grubbs = job3.result()
        ecod_result = job4.result()
        lscp_result = job5.result()
        sr = job6.result()
    
    # zscore = zscore_test(df)
    # f_oneway = oneway_anova_test(df)
    # grubbs = grubbs_test(df)
    # ecod_result = ecod(df)
    # ee = elliptic_envelope(df)
    # lscp_result = lscp(df)
    # sr = spectral_residual_saliency(df)

    cdef dict data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "key": ["标准分数", "单因素方差分析", "Grubbs检验", "ECOD检测", "LSCP检测", "残差谱分析"],
        "value": [zscore, f_oneway, grubbs, ecod_result, lscp_result, sr]
    }
    
    redis_client.hset(f"cv{unit}", "health", json.dumps(data, ensure_ascii=False))

    print(time() - time1)