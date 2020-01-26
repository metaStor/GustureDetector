import cv2
import glob
import os
import time
import config as cfg

'''
将图片压缩成指定大小
图片名不能掺杂中文
'''

scr_path = cfg.IMAGES_PATH
target_path = cfg.IMAGES_PATH

width = 448
height = 448

cate = [os.path.join(scr_path, x) for x in os.listdir(scr_path) if os.path.isdir(os.path.join(scr_path, x))]

# 处理编号
num = 1


def deal_with(file, target):
    global num
    image = cv2.imread(file)
    new_image = cv2.resize(image, (width, height))
    new_file = target + '\\' + str(time.time()) + '.jpg'
    cv2.imwrite(new_file, new_image)
    print('No.' + str(num) + '  Successfully processed in :' + new_file)
    num += 1


# 创建目标路径
if not os.path.exists(target_path):
    os.mkdir(target_path)
    print('成功创建目标路径：' + target_path)

for index, folder in enumerate(cate):
    t1 = folder.split('\\')
    t2 = t1[len(t1) - 1]
    target = os.path.join(target_path, t2)
    if not os.path.exists(target):
        os.mkdir(target)
        print('成功创建目标文件夹路径：' + target)
    for im in glob.glob(folder + '/*.*'):
        temp = im.split('.')
        if temp[len(temp) - 1] == 'jpg' or temp[len(temp) - 1] == 'png':
            deal_with(im, target)
