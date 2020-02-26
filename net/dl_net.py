import config as cfg
import tensorflow as tf
from tensorflow.contrib import slim  # 高级封装框架
import numpy as np


class dl_net(object):

    def __init__(self, is_training=True):
        self.image_size = cfg.IMAGE_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.l2_regularizer = cfg.L2_REGULARIZER
        self.cell_size = cfg.CELL_SIEZ
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.sclae = 1.0 * self.image_size / self.cell_size  # 单位cell的size
        self.boxes_predict_cell = cfg.BOXES_PRE_CELL  # 每个cell内有两个预测的boxes
        # 每个predict_boxes包含5个值： [x轴中点， y轴中点， 宽度， 高度， confidence(置信度)]
        # cell_size*cell_size*(num_class+5*boxes_cell) = 7*7*45(35+10)
        # 21中，11代表预测的类别，10代表2个predict_boxes预测的[x轴中点， y轴中点， 宽度， 高度， confidence(置信度)]
        self.output_size = (self.cell_size * self.cell_size) * \
                           (self.boxes_predict_cell * 5 + self.num_class)
        # 每个cell中的predict_boxes
        # boundery1 指的是对于所有的 cell 的类别的预测的张量维度
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        # boundery2 指的是在类别之后每个cell 所对应的 bounding boxes 的数量的总和
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_predict_cell

        self.class_scale = cfg.CLASS_SCALE
        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.coord_scale = cfg.COORD_SCALE

        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA

        # 输入(7, 7, 2)
        self.offset = np.transpose(
            np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_predict_cell),
                       newshape=(self.boxes_predict_cell, self.cell_size, self.cell_size)), axes=(1, 2, 0))

        # 输入数据, 第一维度是None，表示不限多个数据量，由batch_size决定一次的输入
        self.images = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, 3], name='images')
        # 预测函数
        self.logits = self.build_net(self.images, self.output_size, self.alpha, is_training=is_training)

        if is_training:
            # 真实值
            self.labels = tf.placeholder(tf.float32, shape=[None, self.cell_size, self.cell_size, 5 + self.num_class])
            # 损失函数
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)

    # --------------------------构建网络-------------------------
    def build_net(self,
                  images,
                  num_outputs,
                  alpha,
                  keep_prob=0.5,  # dropout神经元被选中的概率
                  is_training=True,
                  scope='yolo'
                  ):
        with tf.variable_scope(scope):
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected],  # 作用域列表
                    # activation_fn=leaky_relu(alpha),  # 激活函数
                    activation_fn=lambda x: tf.nn.leaky_relu(features=x, alpha=alpha),  # 激活函数
                    weights_regularizer=slim.l2_regularizer(self.l2_regularizer),  # key=value形式
                    weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                    # biases_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            ):
                # images.shape=(none, image_size, image_size, 3), image_size的行和列分别以0扩充3
                # 因为下一个conv2d是VALID模式
                net = tf.pad(tensor=images,
                             paddings=np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
                             mode='CONSTANT',
                             name='pad_1')
                net = slim.conv2d(inputs=net, num_outputs=64,
                                  kernel_size=7, stride=2, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(inputs=net, kernel_size=2, padding='SAME', scope='pool_3')
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                             name='pad_27')
                net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                # 张量扁平化， for example: (None, 4, 4) => (?, 16)
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                # is_training为true的时候，dropout才起作用
                net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training,
                                   scope='dropout_35')
                # 最后一层采用线性激活函数,因为需要预测bounding box的位置（数值型），而不仅仅是对象的概率
                # 输出tensor => (?, 7*7*(35+2*5))
                net = slim.fully_connected(net, num_outputs, activation_fn=None, scope='fc_36')

                # 预测开启
                net = self.interpret_net(net)

        return net

    # 预测时开启
    def interpret_net(self, predict):
        # detect_test
        predict_classes = tf.reshape(predict[:, :self.boundary1],
                                     [-1, self.cell_size, self.cell_size, self.num_class])
        predict_scales = tf.reshape(predict[:, self.boundary1:self.boundary2],
                                    [-1, self.cell_size, self.cell_size, self.boxes_predict_cell])
        predict_boxes = tf.reshape(predict[:, self.boundary2:],
                                   [-1, self.cell_size, self.cell_size, self.boxes_predict_cell * 4])
        result = tf.concat([predict_classes, predict_scales, predict_boxes], 3)

        return result

    # ---------------------------网络结束---------------------------

    def calc_iou(self, predict_boxes, boxes, scope='iou'):
        '''
        :param predict_boxes_tran: shape of tensor: [batch_size, cell_size, cell_size, boxes_pre_cell, 4]
                                                        4 => [x_center, y_center, w, h]
        :param boxes: shape of tensor: [batch_size, cell_size, cell_size, boxes_pre_cell, 4]
                                                        4 => [x_center, y_center, w, h]
        :param scope: calculate ious
        :return: 两个预测box的iou，[batch_size, cell_size, cell_size, boxes_pre_cell]
        '''
        with tf.variable_scope(scope):
            # (x_center, y_center, w, h) ===> (x1, y1, x2, y2)
            # 复原xmin, xmax, ymin, ymax
            # xmin = x_center - (width / 2)
            # ymin = y_center - (height / 2)
            # xmax = x_center + (width / 2)
            # ymax = y_center + (height / 2)
            boxes1_t = tf.stack([predict_boxes[..., 0] - predict_boxes[..., 2] / 2.0,
                                 predict_boxes[..., 1] - predict_boxes[..., 3] / 2.0,
                                 predict_boxes[..., 0] - predict_boxes[..., 2] / 2.0,
                                 predict_boxes[..., 1] - predict_boxes[..., 3] / 2.0],
                                axis=-1)
            boxes2_t = tf.stack([boxes[..., 0] - boxes[..., 2] / 2.0,
                                 boxes[..., 1] - boxes[..., 3] / 2.0,
                                 boxes[..., 0] - boxes[..., 2] / 2.0,
                                 boxes[..., 1] - boxes[..., 3] / 2.0],
                                axis=-1)
            # 计算左上角和右下角的点
            # O ----------→ x
            # |
            # |
            # |
            # |
            # ↓
            # y
            # 左上角点中xmax, ymax
            left_up = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            # 右下角点中xmin, ymin
            right_down = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

            # 交集
            intersection = tf.maximum(0.0, right_down - left_up)
            intersection_square = intersection[..., 0] * intersection[..., 1]

            # 并集, 两个boxes面积之和 - 相交的面积
            predict_boxes_square = predict_boxes[..., 2] * predict_boxes[..., 3]
            boxes_square = boxes[..., 2] * boxes[..., 3]
            union_square = tf.maximum(1e-10, predict_boxes_square + boxes_square - intersection_square)

            # 将一个张量中的数值限制在一个范围之内。（可以避免一些运算错误）
            iou = tf.clip_by_value(intersection_square / union_square, 1e-10, 1.0)

        return iou

    # 将网络输出分离为类别和定位以及box大小
    def loss_layer(self, predict, labels, scope='loss_layer'):
        with tf.variable_scope(scope):
            # 拆分输出结果
            # 类别 [batch_size, 7, 7, 35]
            predict_classes = tf.reshape(
                # 因为输入的images是None， 所以一维是未知数量的，全取
                predict[:, :self.boundary1],
                [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            # 预测boxes的置信度 [batch_size, 7, 7, 2]
            predict_scales = tf.reshape(
                predict[:, self.boundary1:self.boundary2],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_predict_cell])
            # box预测的边界 [batch_size, 7, 7, 2, 4]
            predict_boxes = tf.reshape(
                predict[:, self.boundary2:],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_predict_cell, 4])

            # 拆分真实标签
            # label(真实)的类别 [batch_size, 7, 7, 1]
            response = tf.reshape(
                labels[..., 0],
                [self.batch_size, self.cell_size, self.cell_size, 1])
            # label的定位结果 [batch_size, 7, 7, 1, 4]
            boxes = tf.reshape(
                labels[..., 1:5],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            # label的大小结果， 扩充两倍， 因为有两个预测的box [batch_size, 7, 7, 2, 4]
            # 归一化到[0-1]
            boxes = tf.tile(boxes, multiples=[1, 1, 1, self.boxes_predict_cell, 1]) / self.image_size
            # [batch_size, 7, 7, 35]
            classes = labels[..., 5:]

            # 偏移量 => [1, 7, 7, 2] （2个预测的bounding boxes）
            # 因为cell的坐标只是针对x与y轴的相交处，实际预测中并不都是位于那里
            # 所以要加偏移量，确保它可能落在cell的中心或者中心偏左一点等等
            # 反过来思考，可以将网络输出的预测坐标归一化到[0-1]，在加上偏移量就是真实的坐标了
            offset = tf.reshape(tf.constant(self.offset, dtype=tf.float32),
                                [1, self.cell_size, self.cell_size, self.boxes_predict_cell])
            # 将width所在的轴作为x轴（列），height所在的轴作为y轴（行）
            # x轴的偏移量 [batch_size, cell_size, cell_size, boxes_predict_cell]
            #        => [batch_size, y, x, boxes_predict_cell]
            x_offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            # y轴的偏移量 [batch_size, cell_size, cell_size, boxes_predict_cell]
            #        => [batch_size, y, x, boxes_predict_cell]
            y_offset = tf.transpose(x_offset, (0, 2, 1, 3))
            # 叠加最后一维 [batch_size, 7, 7, 2, 4]
            # bounding box预测的x,y值，是相对与cell左上角坐标的偏移量
            # 也就是，预测输出的（x,y）坐标只是[0-1]的偏移量，要加上offset才是真正的坐标
            predict_boxes_tran = tf.stack(
                [(predict_boxes[..., 0] + x_offset) / self.cell_size,  # x中心，归一化到[0-1]
                 (predict_boxes[..., 1] + y_offset) / self.cell_size,  # y中心，归一化到[0-1]
                 tf.square(predict_boxes[..., 2]),  # w平方
                 tf.square(predict_boxes[..., 3])], axis=-1)  # h平方

            # 得到两个预测boxes的IOU
            # [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # object tensor => [BATCH_SIZE, CELL_SIZE, CELL_SIZE, 1]
            # 找出其中最大的box
            object_mask = tf.reduce_max(iou_predict_truth, axis=3, keep_dims=True)
            # 有物体就为1， 没有就是0
            # 将排除的box全变为0，只剩下最大的box。然后与真实（标注）的相乘。若标注没有物体，即全为0
            object_mask = tf.cast((iou_predict_truth >= object_mask), dtype=tf.float32) * response

            # no_object tensor => [BATCH_SIZE, CELL_SIZE, CELL_SIZE, 1]
            # 将有物体以外的标记为0（1-1=0）
            noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

            # [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]
            # 假设物体的真实中心坐标为（x,y），且其所在的cell左上角坐标为（x_offset, y_offset）
            # 则x轴坐标的偏移量为：（x * cell_size / image_size - x_offset）
            # 且y轴坐标的偏移量为：（y * cell_size / image_size - y_offset）
            boxes_tran = tf.stack(
                [boxes[..., 0] * self.cell_size - x_offset,  # x中心
                 boxes[..., 1] * self.cell_size - y_offset,  # y中心
                 tf.sqrt(boxes[..., 2]),  # w开方
                 tf.sqrt(boxes[..., 3])], axis=-1)  # y开方

            # class loss
            # 类别误差=(预测类别-真实类别)*标注类别
            # 判断是否有object落在cell中
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                name='class_loss') * self.class_scale  # 附加权重（正常）

            # object loss
            # 含有object的box的confidence
            object_dalta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_dalta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale  # 附加权重（正常）

            # noobject loss
            # no有object的box的confidence
            noobject_dalta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_dalta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale  # 附加权重（轻视）

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, axis=4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale  # 附加权重（重视坐标）

            # 总和loss
            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            # 加入日志
            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            # 直方图
            tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            tf.summary.histogram('iou', iou_predict_truth)

# def leaky_relu(alpha):
#     # 返回一个op函数, 用于输入features
#     def op(inputs):
#         return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
#
#     return op
