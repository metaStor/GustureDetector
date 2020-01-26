import os
import config as cfg
import xml.etree.ElementTree as ET

'''
更改大小写
应用于xml文件中的指定字段
'''

source = os.path.join(cfg.DATA_PATH, 'Annotations')

count = 0

for file in sorted(os.listdir(source)):
    xml_file = os.path.join(source, file)
    tree = ET.parse(xml_file)
    obj = tree.find('object')
    name = obj.find('name')
    na = name.text
    if not na.islower() and len(na) == 1:
        name.text = na.lower()
        tree.write(xml_file)
        print('file %s change %s to %s' % (xml_file, na, name.text))
        count += 1

print(count)
