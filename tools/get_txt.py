import os
import config as cfg


class get_txt(object):

    def __init__(self):
        self.path = cfg.IMAGES_PATH
        self.output = os.path.join(cfg.DATA_PATH, 'ImageSets', 'Main')
        self.count = 0
        self.bit = 6

    '''
    得到文件名
    适用于一个文件夹下所有图片已经标号有序的情况
    '''
    def get_filename(self):
        with open(os.path.join(self.output, 'trainval.txt'), 'a+') as f:
            for file in sorted(os.listdir(self.path)):
                name = file.strip().split('.')[0]
                f.write(str(name) + '\n')

    '''
    重命名以及得到文件名
    适用于每个文件夹归属一个类别的情况
    如: a文件夹下, 全为a的图片
    '''
    def rename_and_get_filename(self):
        files = [os.path.join(self.path, x) for x in sorted(os.listdir(self.path))
                 if os.path.isdir(os.path.join(self.path, x))]
        if not os.path.exists(self.output):
            os.makedirs(self.output)
            print('Create ', str(self.output))
        with open(os.path.join(self.output, 'trainval.txt'), 'a+') as f:
            for folder in files:
                for file in sorted(os.listdir(folder)):
                    old = os.path.join(folder, file)
                    lenth = len(str(self.count))
                    add = '0' * (self.bit - lenth)
                    new = os.path.join(folder, str(add) + str(self.count) + '.jpg')
                    # rename
                    os.rename(old, new)
                    # write to trainval.txt
                    f.write(str(add) + str(self.count) + '\n')
                    self.count += 1
            print(self.count)


get_txt().get_filename()
