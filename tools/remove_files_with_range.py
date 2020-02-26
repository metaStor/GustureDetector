#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : remove_files_with_range.py
# @Author: ShenHao
# @Contact : 1427662743@qq.com 
# @Date  : 20-2-26下午7:16
# @Desc  : 根据区间来删除的文件


import os

img_path = r'/media/meta/Work/Study_and_Work/毕业论文/gusture/dataSet/Images'
ann_path = r'/media/meta/Work/Study_and_Work/毕业论文/gusture/dataSet/Annotations'

ranges = (665, 5889)


def rm_with_ranges(path, suffix='jpg'):
    for index in range(ranges[0], ranges[1]):
        lenth = len(str(index))
        add = '0' * (6 - lenth)
        name = str(add) + str(index) + '.' + suffix
        print('Removing %s ...' % name)
        try:
            os.remove(os.path.join(path, name))
        except OSError:
            print('No such file or directory!')
    print('Finished!')


if __name__ == '__main__':
    rm_with_ranges(ann_path, suffix='xml')
    rm_with_ranges(img_path)
