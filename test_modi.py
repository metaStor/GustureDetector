import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer


class Detector(object):

    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file

        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + \
                         self.cell_size * self.cell_size * self.boxes_per_cell

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('Restoring weights from: ' + self.weights_file)
        # 错误写法，猜测只恢复部分变量
        # self.saver = tf.train.import_meta_graph(self.weights_file)
        # self.saver.restore(self.sess, tf.train.latest_checkpoint(cfg.WEIGHTS_DIR))
        self.saver = tf.train.Saver()
        # self.saver.restore(self.sess, self.weights_file)
        # 恢复所有变量
        # 自动获取checkout中的最新模型
        # self.saver.restore(self.sess, tf.train.latest_checkpoint(self.weights_file))
        ckpt = tf.train.get_checkpoint_state(self.weights_file)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def process_predicts1(self, predicts):
        p_classes = predicts[0, :, :, 0:20]
        C = predicts[0, :, :, 20:22]
        coordinate = predicts[0, :, :, 22:]

        p_classes = np.reshape(p_classes, (7, 7, 1, 20))
        C = np.reshape(C, (7, 7, 2, 1))

        P = C * p_classes

        # print P[5,1, 0, :]

        index = np.argmax(P)

        index = np.unravel_index(index, P.shape)

        class_num = index[3]

        coordinate = np.reshape(coordinate, (7, 7, 2, 4))

        max_coordinate = coordinate[index[0], index[1], index[2], :]

        xcenter = max_coordinate[0]
        ycenter = max_coordinate[1]
        w = max_coordinate[2]
        h = max_coordinate[3]

        xcenter = (index[1] + xcenter) * (448 / 7.0)
        ycenter = (index[0] + ycenter) * (448 / 7.0)

        w = w * 448
        h = h * 448

        xmin = xcenter - w / 2.0
        ymin = ycenter - h / 2.0

        xmax = xmin + w
        ymax = ymin + h

        return xmin, ymin, xmax, ymax, class_num

    def detect_test(self, img, wait=0):
        img = cv2.imread(img)
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        predict = self.sess.run(self.net.logits, feed_dict={self.net.images: inputs})

        xmin, ymin, xmax, ymax, class_num = self.process_predicts1(predict)
        class_name = self.classes[class_num]
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
        cv2.putText(img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
        cv2.imshow('detect', img)
        cv2.waitKey(wait)
        self.sess.close()

    def process_predicts2(self, resized_img, predicts, thresh=0.12):
        """
        process the predicts of object detection with one image input.

        Args:
            resized_img: resized source image.
            predicts: output of the model.
            thresh: thresh of bounding box confidence.
        Return:
            predicts_dict: {"cat": [[x1, y1, x2, y2, scores1], [...]]}.
        """
        p_classes = predicts[0, :, :, 0:20]  # 20 classes.
        C = predicts[0, :, :, 20:22]  # two bounding boxes in one cell.
        coordinate = predicts[0, :, :, 22:]  # all bounding boxes position.

        p_classes = np.reshape(p_classes, (7, 7, 1, 20))
        C = np.reshape(C, (7, 7, 2, 1))

        P = C * p_classes  # confidencefor all classes of all bounding boxes (cell_size, cell_size, bounding_box_num, class_num) = (7, 7, 2, 1).

        predicts_dict = {}
        for i in range(7):
            for j in range(7):
                temp_data = np.zeros_like(P, np.float32)
                temp_data[i, j, :, :] = P[i, j, :, :]
                position = np.argmax(
                    temp_data)  # refer to the class num (with maximum confidence) for every bounding box.
                index = np.unravel_index(position, P.shape)

                if P[index] > thresh:
                    class_num = index[-1]
                    coordinate = np.reshape(coordinate, (
                        7, 7, 2, 4))  # (cell_size, cell_size, bbox_num_per_cell, coordinate)[xmin, ymin, xmax, ymax]
                    max_coordinate = coordinate[index[0], index[1], index[2], :]

                    xcenter = max_coordinate[0]
                    ycenter = max_coordinate[1]
                    w = max_coordinate[2]
                    h = max_coordinate[3]

                    xcenter = (index[1] + xcenter) * (448 / 7.0)
                    ycenter = (index[0] + ycenter) * (448 / 7.0)

                    w = w * 448
                    h = h * 448
                    xmin = 0 if (xcenter - w / 2.0 < 0) else (xcenter - w / 2.0)
                    ymin = 0 if (ycenter - h / 2.0 < 0) else (ycenter - h / 2.0)
                    xmax = resized_img.shape[0] if (xmin + w) > resized_img.shape[0] else (xmin + w)
                    ymax = resized_img.shape[1] if (ymin + h) > resized_img.shape[1] else (ymin + h)

                    class_name = self.classes[class_num]
                    predicts_dict.setdefault(class_name, [])
                    predicts_dict[class_name].append([int(xmin), int(ymin), int(xmax), int(ymax), P[index]])

        return predicts_dict

    def non_max_suppress(self, predicts_dict, threshold=0.5):
        """
        implement non-maximum supression on predict bounding boxes.
        Args:
            predicts_dict: {"cat": [[x1, y1, x2, y2, scores1], [...]]}.
            threshhold: iou threshold
        Return:
            predicts_dict processed by non-maximum suppression
        """
        for object_name, bbox in predicts_dict.items():
            bbox_array = np.array(bbox, dtype=np.float)
            x1, y1, x2, y2, scores = bbox_array[:, 0], bbox_array[:, 1], bbox_array[:, 2], bbox_array[:, 3], bbox_array[
                                                                                                             :, 4]
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            # print "areas shape = ", areas.shape
            order = scores.argsort()[::-1]
            # print "order = ", order
            keep = []

            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
                iou = inter / (areas[i] + areas[order[1:]] - inter)
                indexs = np.where(iou <= threshold)[0]
                order = order[indexs + 1]
            bbox = bbox_array[keep]
            predicts_dict[object_name] = bbox.tolist()
            predicts_dict = predicts_dict
        return predicts_dict

    def plot_result(self, src_img, predicts_dict):
        """
        plot bounding boxes on source image.
        Args:
            src_img: source image
            predicts_dict: {"cat": [[x1, y1, x2, y2, scores1], [...]]}.
        """
        height_ratio = src_img.shape[0] / 448.0
        width_ratio = src_img.shape[1] / 448.0
        for object_name, bbox in predicts_dict.items():
            for box in bbox:
                xmin, ymin, xmax, ymax, score = box
                src_xmin = xmin * width_ratio
                src_ymin = ymin * height_ratio
                src_xmax = xmax * width_ratio
                src_ymax = ymax * height_ratio
                score = float("%.3f" % score)

                cv2.rectangle(src_img, (int(src_xmin), int(src_ymin)), (int(src_xmax), int(src_ymax)), (0, 255, 0))
                cv2.putText(src_img, object_name + str(score), (int(src_xmin), int(src_ymin)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0))

        cv2.imshow("result", src_img)
        # cv2.imwrite("result.jpg", src_img)

    def detect_non_max(self, img, wait=0):
        img = cv2.imread(img)
        img_h, img_w, _ = img.shape
        image = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        print('Procession detection...')
        np_predict = self.sess.run(self.net.logits, feed_dict={self.net.images: inputs})

        predicts_dict = self.process_predicts2(image, np_predict)

        predicts_dict = self.non_max_suppress(predicts_dict)

        self.plot_result(img, predicts_dict)
        cv2.waitKey(wait)
        self.sess.close()


# -------------------------------------------------------------


    def draw_result(self, img, result):
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            cv2.putText(
                img, result[i][0] + ' : %.2f' % result[i][5],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType)

    def detect(self, img):
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        result = self.detect_from_cvmat(inputs)[0]

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)

        return result

    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        results = []
        # for i in net_output[0]:
        #     print(i)
        # assert False
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

        return results

    def interpret_output(self, output):
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.num_class))
        class_probs = np.reshape(
            output[0:self.boundary1],
            (self.cell_size, self.cell_size, self.num_class))
        scales = np.reshape(
            output[self.boundary1:self.boundary2],
            (self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = np.reshape(
            output[self.boundary2:],
            (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        offset = np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell)
        offset = np.transpose(
            np.reshape(
                offset,
                [self.boxes_per_cell, self.cell_size, self.cell_size]),
            (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= self.image_size

        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])

        # print(probs)
        # assert False

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(
            filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append(
                [self.classes[classes_num_filtered[i]],
                 boxes_filtered[i][0],
                 boxes_filtered[i][1],
                 boxes_filtered[i][2],
                 boxes_filtered[i][3],
                 probs_filtered[i]])

        return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
             max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
             max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

    def camera_detector(self, cap, wait=10):
        detect_timer = Timer()
        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            detect_timer.tic()
            result = self.detect(frame)
            detect_timer.toc()
            print('Average detecting time: {:.3f}s'.format(
                detect_timer.average_time))

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)

            ret, frame = cap.read()

    def image_detector(self, imname, wait=0):
        # print(tf.get_default_graph().get_tensor_by_name('variables:0'))
        # values = tf.trainable_variables()
        # [print(x) for x in values]
        # print(self.sess.run('yolo/conv_4/biases:0'))
        # assert False
        detect_timer = Timer()
        image = cv2.imread(imname)

        detect_timer.tic()
        result = self.detect(image)
        detect_timer.toc()
        print('Average detecting time: {:.3f}s'.format(
            detect_timer.average_time))

        self.draw_result(image, result)
        cv2.imshow('Image', image)
        cv2.waitKey(wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo = YOLONet(False)
    # weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)
    weight_file = cfg.WEIGHTS_DIR
    detector = Detector(yolo, weight_file)

    # detect from camera
    # cap = cv2.VideoCapture(0)
    # detector.camera_detector(cap)

    # detect from image file
    # imname = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit', 'VOC2007+2012', 'JPEGImages', '000005.jpg')
    imname = './test/person.jpg'
    # detector.detect_test(imname, 5000)
    detector.detect_non_max(imname, 5000)
    # detector.image_detector(imname, 5000)


if __name__ == '__main__':
    main()
