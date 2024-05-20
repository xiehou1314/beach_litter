import os, cv2

import numpy
import numpy as np
from cv2.typing import Scalar

# img_base_path = '../../datasets/Beach Plastic Litter Dataset version 1/plastic_coco/images/train'
# lab_base_path = '../../datasets/Beach Plastic Litter Dataset version 1/plastic_coco/labels/train'
img_base_path = '../../datasets/coco/images/train2017'
lab_base_path = '../../datasets/coco/labels/train2017'

label_path_list = [i.split('.')[0] for i in os.listdir(img_base_path)]
for path in label_path_list:
    image = cv2.imread(f'{img_base_path}/{path}.jpg')
    h, w, c = image.shape
    label = np.zeros((h, w), dtype=np.uint8)
    with open(f'{lab_base_path}/{path}.txt') as f:
        mask = np.array(list(map(lambda x:np.array(x.strip().split()), f.readlines())),dtype=object)
    for i in mask:
        i = np.array(i, dtype=np.float32)[1:].reshape((-1, 2))
        vertexs = np.empty([4,2],dtype=np.float32)
        i[:, 0] *= w
        i[:, 1] *= h
        center_x = i[0][0]
        center_y = i[0][1]
        half_w = i[1][0]/2
        half_h = i[1][1]/2
        vertex_1 =[center_x-half_w,center_y-half_h]
        vertex_2=[center_x+half_w,center_y-half_h]
        vertex_3=[center_x+half_w,center_y+half_h]
        vertex_4=[center_x-half_w,center_y+half_h]
        vertexs[0]=vertex_1
        vertexs[1]=vertex_2
        vertexs[2]=vertex_3
        vertexs[3]=vertex_4
        label = cv2.fillPoly(label, [np.array(vertexs, dtype=np.int32)], color=(255,0,0),lineType=1)
    image = cv2.bitwise_and(image, image, mask=label)
    cv2.imshow('Pic', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

