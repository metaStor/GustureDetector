import tensorflow as tf
from tensorflow.contrib import slim
import os
import datetime
import config as cfg
from net.dl_net import dl_net
from util.gesture import gesture
from util.timer import Timer


class Solver(object):

    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.batch_size = cfg.BATCH_SIZE
        self.wight_path = cfg.WEIGHT_PATH
        self.wight_file = cfg.WEIGHT_FILE
        self.max_iter = cfg.MAX_ITER
        self.save_iter = cfg.SAVE_ITER
        self.summary_iter = cfg.SUMMARY_ITER
        self.init_learning_rate = cfg.LEARNING_RATE
        self.decay_rate = cfg.DECAY_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.staircase = cfg.STAIRCASE
        self.output_path = os.path.join(
            cfg.OUTPUT_PATH, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print('Create the output dir in: ', str(self.output_path))
        self.save_config()

        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.ckpt_file = os.path.join(self.output_path, 'gusture')
        # 用于tensorboard可视化网络
        # 保存到硬盘中
        self.summary_op = tf.summary.merge_all()
        # 写入到指定文件
        self.writer = tf.summary.FileWriter(self.output_path, flush_secs=60)

        # 创建全局tensor
        self.global_step = tf.train.create_global_step()
        # 学习速率指数衰减法
        # 先使用较大的学习率来快速得到一个比较优的解，然后随着迭代的继续逐步减小学习率
        # 随着迭代次数的增加，学习率逐步降低，使得模型更加稳定
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=self.init_learning_rate,
            global_step=self.global_step,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=self.staircase,
            name='learning_rate')
        # 构造学习器， 梯度下降法
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate, name='GradientDescent')
        # Create the train_op
        self.train_op = slim.learning.create_train_op(
            total_loss=self.net.total_loss, optimizer=self.optimizer, global_step=self.global_step)

        gpu_options = tf.GPUOptions()
        # 分配的显存比例
        gpu_options.per_process_gpu_memory_fraction = 0.7
        config = tf.ConfigProto(
            # device_count={'gpu': 0},  # 使用第一块GPU
            gpu_options=gpu_options)
        config.gpu_options.allow_growth = True  # 需要多少用多少
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        # restoring weight file
        if self.wight_file is not None:
            print('Restoring weight file for ', self.wight_file)
            self.saver.restore(self.sess, self.wight_file)

        self.writer.add_graph(graph=self.sess.graph)

    def train(self):

        load_time = Timer()
        train_time = Timer()

        for step in range(1, self.max_iter + 1):
            # load images
            load_time.tic()
            images, labels = self.data.get()
            load_time.toc()

            feed_dict = {self.net.images: images,
                         self.net.labels: labels}

            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:
                    train_time.tic()
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.total_loss, self.train_op],
                        feed_dict=feed_dict)
                    train_time.toc()
                    log_str = '''{} Epoch: {}, step: {}, learning rate: {},
                              Loss: {}\nSpeed: {}s/iter, Load: {}s/iter， 
                              Remain: {}'''.format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        self.data.epoch,
                        int(step),
                        round(self.learning_rate.eval(session=self.sess), 6),
                        loss,
                        train_time.average_time,
                        load_time.average_time,
                        train_time.remain(step, self.max_iter))
                    print(log_str)
                else:
                    train_time.tic()
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op], feed_dict=feed_dict)
                    train_time.toc()
                # 保存训练记录和步数，画图记录
                self.writer.add_summary(summary_str, step)
            else:
                train_time.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_time.toc()
            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'), self.output_path))
                self.saver.save(self.sess, self.ckpt_file, global_step=self.global_step)

    # save the configuration
    def save_config(self):
        with open(os.path.join(self.output_path, 'config.txt'), 'w') as f:
            dict_cfg = cfg.__dict__
            for key in sorted(dict_cfg.keys()):
                if key[0].isupper():
                    info = '{}: {}\n'.format(key, dict_cfg[key])
                    f.write(info)


def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    net = dl_net()

    data = gesture(phase='train', rebuild=False)

    solver = Solver(net=net, data=data)

    print('Start training...')
    solver.train()
    print('Done training...')


if __name__ == '__main__':
    main()
