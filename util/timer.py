import time
import datetime


class Timer(object):

    def __init__(self):
        self.init_time = time.time()
        self.total_time = 0
        self.start_time = 0
        self.times = 0  # 次数
        self.diff = 0
        self.average_time = 0
        self.remain_time = 0

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.times += 1
        self.average_time = self.total_time / self.times
        if average:
            return self.average_time
        else:
            return self.diff

    # 计算大约剩余时间
    def remain(self, iters, max_iters):
        if iters == 0:
            self.remain_time = 0
        else:
            # 当前iters花的时间*（剩余iters次数/当前iters）
            self.remain_time = (time.time() - self.init_time) * \
                               (max_iters - iters) / iters
        return str(datetime.timedelta(seconds=int(self.remain_time)))
