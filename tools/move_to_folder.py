import os
import shutil
import config as cfg

'''
将当前目录下的所有文件夹中的文件复制到当前文件
'''

source = cfg.ANNOTATIONS

target = cfg.ANNOTATIONS

count = 0

folder_files = []

files = []

for folder in os.listdir(source):
    for file in os.listdir(os.path.join(source, folder)):
        if file not in files:
            folder_files.append(str(folder) + ':' + str(file))
            files.append(file)
            # 移动文件
            source_file = os.path.join(source, folder, file)
            print('Move %s to %s ' % (source_file, target))
            shutil.move(src=source_file, dst=target)
        else:
            # 输出重复文件
            for tmp in folder_files:
                t = tmp.split(':')
                if t[1] == file:
                    print('file: %s and file: %s is conflict' %
                          (os.path.join(source, folder, file),
                           os.path.join(source, t[0], t[1])))
        count += 1
print(count)

# 删除空文件夹
for folder in os.listdir(source):
    dirs = os.path.join(source, folder)
    if os.path.isdir(dirs):
        # if len(os.listdir(os.path.join(source, folder))):
        if not os.listdir(dirs):
            print('delete ', str(dirs))
            os.rmdir(dirs)
