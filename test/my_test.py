from skimage import io, transform
import tensorflow as tf
import numpy as np
from test import voice
import tkinter as tk
import cv2
import threading
import time


def target_start(event):
    threading.Thread(target=start).start()


def work():
    model_path = r'D:\DM help\test_data1\model.ckpt.meta'
    scr_path = r'D:/DM help/weight/'

    path1 = r"D:\DM help\temp\1.jpg"
    path2 = r"D:\DM help\temp\2.jpg"
    path3 = r"D:\DM help\temp\3.jpg"
    path4 = r"D:\DM help\temp\4.jpg"
    path5 = r"D:\DM help\temp\5.jpg"

    gesture_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

    w = 80
    h = 80
    c = 3

    def read_one_image(path):
        img = io.imread(path)
        img = transform.resize(img, (w, h))
        return np.asarray(img)

    with tf.Session() as sess:
        data = []
        data1 = read_one_image(path1)
        data2 = read_one_image(path2)
        data3 = read_one_image(path3)
        data4 = read_one_image(path4)
        data5 = read_one_image(path5)

        data.append(data1)
        data.append(data2)
        data.append(data3)
        data.append(data4)
        data.append(data5)

        # 复原模型
        saver = tf.train.import_meta_graph(model_path)
        # tf.train.latest_checkpoint 自动获取最后一次保存的模型
        model_file = tf.train.latest_checkpoint(scr_path)
        # 最新版本的restore需要指定目录（因一个目录下有多个文件）
        saver.restore(sess, model_file)

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x: data}

        logits = graph.get_tensor_by_name("logits_eval:0")

        classification_result = sess.run(logits, feed_dict)

        # 打印出预测矩阵
        print(classification_result)
        # 打印出预测矩阵每一行最大值的索引
        print(tf.argmax(classification_result, 1).eval())
        # 根据索引通过字典对应花的分类
        output = []
        output = tf.argmax(classification_result, 1).eval()
        # 保存类比结果
        result = []
        for i in range(len(output)):
            # print("第", i + 1, "个手势预测:" + gesture_dict[output[i]])
            result.append(gesture_dict[output[i]])
        # 统计
        a = {}
        for x in result:
            a[x] = result.count(x)
        # 返回最大值
        max_value = 0
        value = ''
        for k in a.keys():
            if a.get(k) > max_value:
                max_value = a.get(k)
                value = k
        print('手势预测： ' + value)
        t.delete(0.0, tk.END)
        t.insert(index=0.0, chars=value + '\n')
        # 播放语音
        voice.sound(value)
        sess.close()
        return value


def start():
    time_max = 20
    max_count = 5
    path = r'D:\DM help\temp'
    cap = cv2.VideoCapture(1)
    count = 1  # 抓取图片的数量
    t = 1
    isenough = False
    # 准备时间
    time.sleep(1.5)
    while True:
        # get a image
        _, image = cap.read()
        if t % time_max == 0 and not isenough:
            if count == 3:
                # show the image
                cv2.imshow('Target', image)
            # save image
            cv2.imwrite(path + '\\' + str(count) + '.jpg', image)
            count += 1
            if count > max_count:
                isenough = True
                # working
                threading.Thread(target=work).start()
        # show a frame
        cv2.imshow("Camera", image)
        # quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        t += 1
    cap.release()
    cv2.destroyAllWindows()


window = tk.Tk()
window.title('显示台')
window.geometry('300x420')
window.resizable(width=False, height=False)  # 不可变化窗口大小

# 结果区
result = tk.Frame(window)
t = tk.Text(result, font='Arial, 50', width=300, height=5)
# t.config(state=tk.DISABLED)  # 不可编辑
t.pack(side=tk.TOP)
result.pack(padx=20, side=tk.RIGHT)

# 按钮区
bstart = tk.Button(result, text="开始", width=15, height=8, bg='Snow', relief='groove')
bstart.pack(padx=5, pady=20, side=tk.TOP)
bstart.bind('<Button-1>', func=target_start)
window.mainloop()
