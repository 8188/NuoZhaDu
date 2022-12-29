import schedule 
# import sched
# import time
import multiprocessing
from utils import (
    calculate_open_close_time,
    calculate_pump_run_time,
    health,
)
import warnings
warnings.filterwarnings('ignore')


units = list(range(1,10))


def task1():
    # print("task1:")
    pools = multiprocessing.Pool(9)
    pools.map(calculate_open_close_time, units)

    
def task2():
    # print("task2:")
    pools = multiprocessing.Pool(9)
    pools.map(calculate_pump_run_time, units)


def task3():
    # print("task3:")
    pools = multiprocessing.Pool(9)
    pools.map(health, units)


if __name__ == "__main__":
    # s = sched.scheduler(time.time, time.sleep)
    # while 1:
    #     s.enter(60, 1, task3)
    #     s.run()
    
    schedule.every(1).minutes.do(task3)
    while 1: 
        schedule.run_pending() 
