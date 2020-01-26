import config as cfg
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import pickle
import copy


class gesture(object):

    def __init__(self, phase, rebuild=False):
        self.data_path = cfg.DATA_PATH
        self.cache_path = cfg.CACHE_PATH
        self.weight_path = cfg.WEIGHT_PATH
        self.image_size = cfg.IMAGE_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.cell_size = cfg.CELL_SIEZ
        self.num_class = len(cfg.CLASSES)
        self.classes_ind = dict(zip(cfg.CLASSES, range(self.num_class)))
        self.flipped = cfg.FLIPPED
        self.phase = phase
        self.rebuild = rebuild
        self.cursor = 0  # 标记当前cell
        self.epoch = 1  # 迭代次数
        self.image_index = None
        self.gt_labels = None
        self.prepare()

    def get(self):
        images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros((self.batch_size, self.cell_size, self.cell_size, (4 + self.num_class + 1)))
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.read_img(imname, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            # 读取完一遍之后打乱顺序
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    # 读取图片
    def read_img(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image, dsize=(self.image_size, self.image_size))
        # cv2默认为 BGR顺序，而其他软件一般使用RGB，所以需要转换
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # 数据归一化， 采用min-max标准化：y = 2*(x-min)/(max-min)-1
        image = (image / 255.0) * 2.0 - 1.0
        # 翻转目标中心
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self):
        gt_labels = self.load_labels()
        if self.flipped:
            # 附加水平翻转训练实例
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)  # copy with sub_list
            for index in range(len(gt_labels_cp)):
                # 确认水平翻转
                gt_labels_cp[index]['flipped'] = True
                # 水平翻转数组(利用步长-1)
                gt_labels_cp[index]['label'] = gt_labels_cp[index]['label'][:, ::-1, :]
                # 所有目标中点水平迁移
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[index]['label'][i, j, 0] == 1:
                            # 物体中心坐标镜像
                            gt_labels_cp[index]['label'][i, j, 1] = \
                                self.image_size - 1 - gt_labels_cp[index]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        # 打乱数据集, 多维默认第一个维度
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    # 加载图片label
    def load_labels(self):

        cache_file = os.path.join(self.cache_path, 'pascal_' + self.phase + '_gt_label.pkl')

        # 恢复训练的数据文件, 只读取一次就好
        if os.path.isfile(cache_file) and not self.rebuild:
            print('Restore gt_labels file from %s ' % str(cache_file))
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
            print('Create cache dir...')

        if self.phase == 'train':
            txt_name = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        else:
            txt_name = os.path.join(self.data_path, 'ImageSets', 'Main', 'test.txt')
        with open(txt_name, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]

        gt_labels = []
        for index in self.image_index:
            # tip
            print("Loading the %s annotation..." % os.path.join(self.data_path, 'Images', index + '.jpg'))
            label, num = self.load_annotations(index)
            if num == 0:
                continue
            imname = os.path.join(self.data_path, 'Images', index + '.jpg')
            gt_labels.append({
                'imname': imname,
                'label': label,
                'flipped': False
            })
        print('Saving gt_labels to ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

    # 加载Annotations标注
    def load_annotations(self, index):
        # XML file of the PASCAL_VOC format
        imname = os.path.join(self.data_path, 'Images', index + '.jpg')
        im = cv2.imread(imname)
        h_rate = 1.0 * im.shape[0] / self.image_size
        w_rate = 1.0 * im.shape[1] / self.image_size
        # im = cv2.resize(im, [self.image_size, self.image_size])
        '''
        (7, 7, 16)
        7*7代表把图片分成7*7个cell， 40表示35个分类+4个boxes坐标(由于数组左闭右开，所以是49+1)
        其中的(7, 7)定位图片中心，
        (40)中0下标代表是否有目标物体， 1～4代表boxes信息，5～39中代表目标是哪一类
        '''
        label = np.zeros((self.cell_size, self.cell_size, (4 + self.num_class + 1)))
        xmlname = os.path.join(cfg.DATA_PATH, 'Annotations', index + '.xml')
        tree = ET.parse(xmlname)
        objs = tree.findall('object')

        # 寻找里面的object
        for obj in objs:
            # object位置
            bbox = obj.find('bndbox')
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_rate, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_rate, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_rate, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_rate, self.image_size - 1), 0)
            # 分类下标
            class_ind = self.classes_ind[obj.find('name').text.lower().strip()]
            # boxes， 包含[x中心点， y中心点， 宽度， 高度]
            boxes = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]
            # 中心点在7*7cell中的位置
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + class_ind] = 1
        # 返回信息和有多少类目标
        return label, len(objs)
