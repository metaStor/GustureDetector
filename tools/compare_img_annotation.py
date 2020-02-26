#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : compare_img_annotation.py
# @Author: ShenHao
# @Contact : 1427662743@qq.com 
# @Date  : 20-2-26下午6:59
# @Desc  : 比较Annotation文件和image文件


import os

img_path = r'/media/meta/Work/Study_and_Work/毕业论文/gusture/DATA'
ann_path = r'/media/meta/Work/Study_and_Work/毕业论文/gusture/ANNOTATIONS'


def load_name(path):
    files = []
    for file in sorted((os.listdir(path))):
        p1 = os.path.join(path, file)
        if os.path.isdir(p1):
            for f in sorted((os.listdir(p1))):
                p2 = os.path.join(p1, f)
                name1 = f.split('.')[0]
                if name1 not in files:
                    files.append(name1)
        else:
            name2 = file.split('.')[0]
            if name2 not in files:
                files.append(name2)
    return files


if __name__ == '__main__':
    imgs = load_name(img_path)
    anns = load_name(ann_path)
    print('Images: %s' % len(imgs))
    print('Annotations: %s' % len(anns))
    print('Find diff files: ')
    for t in imgs:
        if t not in anns:
            print(t, end='')
