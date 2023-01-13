
import numpy as np
import time
import sys
import torch
import os
import random


# def c():
#     '''
#     设置整个开发环境的seed
#     '''
#     seed = 1234
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed) #numpy
#     random.seed(seed) #random and transforms

#     torch.manual_seed(seed) # cpu
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed) #gpu
#         torch.cuda.manual_seed_all(seed) #all gpus
#         # some cudnn methods can be random even after fixing the seeed unless you tell it to be deterministic 
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic=True # cudnn
#         torch.backends.cudnn.enabled = True
#         # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
#         # os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
#         torch.use_deterministic_algorithms(True)
#     return seed

def seed_everything(seed: int = 7):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    # # # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 使用gpu时若出错可以显示出错的具体地方，用了之后程序会卡住不动
    # # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'  # cuda版本大于10.2时，有些函数可能会有不确定过程，因此要加上这个
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = True
    # torch.use_deterministic_algorithms(True) # 检测是否使用了随机算法，有使用随机算法就会报错，你需要一一解决
    print(f"Random seed set as {seed}")
    return seed

# seed = seed_everything()

class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "w")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v


class Timer:
    def __init__(self, starting_msg = None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))


    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        self.est_finish = int(self.start + self.est_total)


    def str_est_finish(self):
        return str(time.ctime(self.est_finish))

    def get_stage_elapsed(self):
        return time.time() - self.stage_start

    def reset_stage(self):
        self.stage_start = time.time()


from multiprocessing.pool import ThreadPool

class BatchThreader:

    def __init__(self, func, args_list, batch_size, prefetch_size=4, processes=12):
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size

        self.pool = ThreadPool(processes=processes)
        self.async_result = []

        self.func = func
        self.left_args_list = args_list
        self.n_tasks = len(args_list)

        # initial work
        self.__start_works(self.__get_n_pending_works())


    def __start_works(self, times):
        for _ in range(times):
            args = self.left_args_list.pop(0)
            self.async_result.append(
                self.pool.apply_async(self.func, args))


    def __get_n_pending_works(self):
        return min((self.prefetch_size + 1) * self.batch_size - len(self.async_result)
                   , len(self.left_args_list))

    # 通过模拟卷积操作强制cudnn初始化
    def __force_cudnn_initialization(self):
        s = 32
        dev = torch.device('cuda')
        torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

    def pop_results(self):

        n_inwork = len(self.async_result)

        n_fetch = min(n_inwork, self.batch_size)
        self.__force_cudnn_initialization()
        rtn = [self.async_result.pop(0).get()
                for _ in range(n_fetch)]

        to_fill = self.__get_n_pending_works()
        if to_fill == 0:
            self.pool.close()
        else:
            self.__start_works(to_fill)

        return rtn




def get_indices_of_pairs(radius, size):

    search_dist = []

    for x in range(1, radius):
        search_dist.append((0, x))

    for y in range(1, radius):
        for x in range(-radius + 1, radius):
            if x * x + y * y < radius * radius:
                search_dist.append((y, x))

    radius_floor = radius - 1

    full_indices = np.reshape(np.arange(0, size[0]*size[1], dtype=np.int64),
                                   (size[0], size[1]))

    cropped_height = size[0] - radius_floor
    cropped_width = size[1] - 2 * radius_floor

    indices_from = np.reshape(full_indices[:-radius_floor, radius_floor:-radius_floor],
                              [-1])

    indices_to_list = []

    for dy, dx in search_dist:
        indices_to = full_indices[dy:dy + cropped_height,
                     radius_floor + dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])

        indices_to_list.append(indices_to)

    concat_indices_to = np.concatenate(indices_to_list, axis=0)

    return indices_from, concat_indices_to

def get_indices_of_pairs_circle(radius, size):

    search_dist = []

    for y in range(-radius + 1, radius):
        for x in range(-radius + 1, radius):
            if x * x + y * y < radius * radius and x*x+y*y!=0:
                search_dist.append((y, x))

    radius_floor = radius - 1

    full_indices = np.reshape(np.arange(0, size[0]*size[1], dtype=np.int64),
                                   (size[0], size[1]))

    cropped_height = size[0] - 2 * radius_floor
    cropped_width = size[1] - 2 * radius_floor

    indices_from = np.reshape(full_indices[radius_floor:-radius_floor, radius_floor:-radius_floor],
                              [-1])

    indices_to_list = []

    for dy, dx in search_dist:
        indices_to = full_indices[radius_floor + dy : radius_floor + dy + cropped_height,
                                  radius_floor + dx : radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])

        indices_to_list.append(indices_to)

    concat_indices_to = np.concatenate(indices_to_list, axis=0)

    return indices_from, concat_indices_to
