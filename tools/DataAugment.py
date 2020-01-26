import os
import copy
import cv2
import time

'''
水平镜像翻转
增加数据量
'''

scr_path = r'F:\DM help - 05\dataSet\target'
save_path = r'F:\DM help - 05\dataSet\temp'

for file_list in os.listdir(scr_path):
    cur_path = os.path.join(scr_path, file_list)
    cur_save = os.path.join(save_path, file_list)
    # 创建新生成的保存目录
    if not os.path.exists(cur_save):
        print('创建目标路径...' + cur_save)
        os.mkdir(cur_save)
    else:
        print('已存在目标路径...' + cur_save)

    # 记录处理个数
    k = 0

    # 开始水平镜像
    for file in os.listdir(cur_path):
        file_name = os.path.join(cur_path, file)
        file_name = file_name.replace('\\', '/')  # 防止转义
        print('处理编号' + str(k) + '   ' + file_name + '....')
        suff = file_name.split('.')[1]  # 获取文件后缀
        image = cv2.imread(file_name)
        height = image.shape[0]
        width = image.shape[1]
        new_image = copy.deepcopy(image)
        for i in range(height):
            for j in range(width):
                new_image[i, width - j - 1] = image[i, j]
        new_name = cur_save + '\\' + str(time.time()) + '.' + suff
        cv2.imwrite(new_name, new_image)
        print('成功生成  ' + new_name)
        k += 1

# cv2.imshow('12', image)
# cv2.waitKey(0)
