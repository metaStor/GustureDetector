import os

DATA_PATH = '/media/meta/Work/Study_and_Work/毕业论文/GustureDetector/dataSet'

ANNOTATIONS = os.path.join(DATA_PATH, 'Annotations')

IMAGES_PATH = os.path.join(DATA_PATH, 'Images')

CACHE_PATH = os.path.join(DATA_PATH, 'cache')

OUTPUT_PATH = os.path.join(DATA_PATH, 'output')

WEIGHT_PATH = os.path.join(DATA_PATH, 'weight')

WEIGHT_DIR = os.path.join(OUTPUT_PATH, '2018_12_25_22_15')

WEIGHT_FILE = None
# WEIGHT_FILE = os.path.join(WEIGHT_PATH, 'YOLO_small.ckpt')
# WEIGHT_FILE = os.path.join(WEIGHT_DIR, 'gusture-3000.meta')

# 35 classes
CLASSES = ['zero', 'one', 'two', 'three', 'four', 'five',
           'six', 'seven', 'eight', 'nine', 'a', 'b', 'c', 'd',
           'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p',
           'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

FLIPPED = True  # 水平翻转训练

IMAGE_SIZE = 448

CELL_SIEZ = 7

BOXES_PRE_CELL = 2

ALPHA = 0.1

'''
由于简单粗暴的全部采用了sum-squared error loss
4维的localization error和35维的classification error同等重要显然是不合理的 
另外，每一张图像中，很多 grid cells 并没不包含物体，使得这些 cells 的 confidence 置为 0，
这些不包含物体的 grid cells 的梯度更新，将会以压倒性的优势，覆盖掉包含物体的 grid cells 进行的梯度更新。
这些问题会使得模型不稳定，甚至造成网络的发散。
最主要的就是怎么设计损失函数，让坐标误差，分类误差，iou误差三个方面得到很好的平衡
'''
OBJECT_SCALE = 1.0  # iou分类误差

NOOBJECT_SCALE = 0.5

CLASS_SCALE = 1.0

COORD_SCALE = 5.0  # 增加坐标误差权重比例（更重视坐标预测）

# TRAIN PARAMETER

GPU = '0'  # 第一块显卡
# GPU = '0,1'  # 第一,二块显卡
# GPU = '-1'  # 不使用GPU

LEARNING_RATE = 0.0001

L2_REGULARIZER = 0.0005

BATCH_SIZE = 16

MAX_ITER = 10000

SUMMARY_ITER = 10

SAVE_ITER = 2000

# decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
DECAY_RATE = 0.1  # 衰减系数

# 控制衰减速度
# 如果decay_steps大一些,(global_step / decay_steps)就会增长缓慢一些
# 从而指数衰减学习率decayed_learning_rate就会衰减得慢一些
# 否则学习率很快就会衰减为趋近于0
DECAY_STEPS = 3000

# 以不连续的间隔衰减学习速率，最后曲线就是锯齿状
STAIRCASE = False

# TEST PARAMETER

THRESHOLD = 0.2  # 阈值

IOU_THRESHOLD = 0.5  # IOU阈值
