from curses import keyname
import xml.etree.ElementTree as ET
import os
import cv2
import shutil

VOC_CLASSES = (# always index 0
    'hand')

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    print(filename)
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        difficult = int(obj.find('difficult').text)
        if difficult == 1:
            # print(filename)
            continue
        obj_struct['name'] = obj.find('name').text
        if obj_struct['name'] == 'hand':
        # obj_struct['pose'] = obj.find('pose').text
        # obj_struct['truncated'] = int(obj.find('truncated').text)
        # obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                                int(float(bbox.find('ymin').text)),
                                int(float(bbox.find('xmax').text)),
                                int(float(bbox.find('ymax').text))]
            objects.append(obj_struct)

    return objects


txt_file = open('./dataset/hand_dataset_1.txt', 'w')
# test_file = open('voc07testimg.txt', 'r')
# lines = test_file.readlines()
# lines = [x[:-1] for x in lines]
# print(lines)

Annotations = '/home/xcy/my_code/seg/dataset/Annotations/'
xml_files = os.listdir(Annotations)

count = 0
for xml_file in xml_files:
    count += 1
    # if xml_file.split('.')[0] not in lines:
    #     # print(xml_file.split('.')[0])
    #     continue
    image_path = xml_file.split('.')[0] + '.jpg'
    img=cv2.imread(os.path.join('/home/xcy/my_code/seg/VOC2007/JPEGImages',image_path))
    print(count)
    results = parse_rec(Annotations + xml_file)
    if len(results) == 0:
        print(xml_file)
        continue
    txt_file.write(image_path)
    # num_obj = len(results)
    # txt_file.write(str(num_obj)+' ')
    for result in results:
        class_name = result['name']
        bbox = result['bbox']
        class_name = VOC_CLASSES.index(class_name)
        txt_file.write(' ' +
                       str(bbox[0]) +
                       ' ' +
                       str(bbox[1]) +
                       ' ' +
                       str(bbox[2]) +
                       ' ' +
                       str(bbox[3]) +
                       ' ' +
                       str(class_name))
        cv2.rectangle(img,pt1=(bbox[0],bbox[1]),pt2=(bbox[2],bbox[3]),color=(255,0,0),thickness=2)
    txt_file.write('\n')
              
    # if key==97:
        # print("saved")
        # shutil.copy(Annotations + xml_file,'/home/xcy/my_code/seg/dataset/Annotations/'+ xml_file)
        # cv2.imwrite(os.path.join('/home/xcy/my_code/seg/dataset/img/',xml_file.split('.')[0]+'k.jpg'),img)
    shutil.copy(os.path.join('/home/xcy/my_code/seg/VOC2007/JPEGImages',image_path),'/home/xcy/my_code/seg/dataset/img/'+ image_path)
    # if count == 10:
    #    break
txt_file.close()