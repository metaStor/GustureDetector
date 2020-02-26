# 　测试GPU是否工作

import tensorflow as tf
import timeit

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.device('/cpu:0'):
    random_image_cpu = tf.random_normal((100, 100, 100, 3))
    net_cpu = tf.layers.conv2d(random_image_cpu, 32, 7)
    net_cpu = tf.reduce_sum(net_cpu)

with tf.device('/device:GPU:0'):
    random_image_gpu = tf.random_normal((100, 100, 100, 3))
    net_gpu = tf.layers.conv2d(random_image_gpu, 32, 7)
    net_gpu = tf.reduce_sum(net_gpu)

sess = tf.Session(config=config)

# 确保TF可以检测到GPU
try:
    sess.run(tf.global_variables_initializer())
except tf.errors.InvalidArgumentError:
    print(
        'nn此错误很可能表示此笔记本未配置为使用GPU。 '
        '通过命令面板（CMD/CTRL-SHIFT-P）或编辑菜单在笔记本设置中更改此设置.nn')
    raise


def cpu():
    sess.run(net_cpu)


def gpu():
    sess.run(net_gpu)


# 运行一次进行测试
cpu()
gpu()

# 多次运行op
print('将100*100*100*3通过滤波器卷积到32*7*7*3(批处理x高度x宽度x通道)大小的图像'
      '计算10次运训时间的总和')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU加速超过CPU: {}倍'.format(int(cpu_time / gpu_time)))

sess.close()
