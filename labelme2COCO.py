# -*- coding:utf-8 -*-
# !/usr/bin/env python

import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from labelme import utils
import numpy as np
import glob
import PIL.Image

class labelme2coco(object):

    def __init__(self,labelme_json=[],save_json_path='./train_val_new.json'):
        '''
        :param labelme_json: #所有labelme的json文件路径组成的列表
        :param save_json_path: #json保存位置
        '''
        self.labelme_json=labelme_json
        self.save_json_path=save_json_path
        self.images=[]
        self.categories=[]
        self.categories.append(self.categorie('person'))
        self.annotations=[]
        # self.data_coco = {}
        self.label=[]
        self.annID=3000000
        self.height=0
        self.width=0

        self.save_json()

    def data_transfer(self):
        for num,json_file in enumerate(self.labelme_json):
            print('jsonfile: ',json_file)
            with open(json_file,'r') as fp:
                data = json.load(fp)  # 加载json文件
                self.images.append(self.image(data,num))
                self.kLabelInfo = []
                if 'keypoints' in data.keys():
                    for index,keypoints in enumerate(data['keypoints']):
                        # print(keypoints)
                        self.pointsNum = 0
                        self.keypointList = []
                        for points in keypoints['points']:
                            # print(points)
                            if(points[2] == 0):
                                points[0] = 0
                                points[1] = 0
                                self.keypointList.append(points[0])
                                self.keypointList.append(points[1])
                                self.keypointList.append(points[2])
                                continue
                            self.pointsNum +=1
                            self.keypointList.append(points[0])
                            self.keypointList.append(points[1])
                            self.keypointList.append(points[2])
                            # print((data['keypoints'])['points'])
                        self.kLabelInfo.append([keypoints['label'],self.keypointList,self.pointsNum])
                if 'shapes' in data.keys():
                    for shapes in data['shapes']:
                        label=shapes['label'].split('_')
                        # if label[1] not in self.label:
                            # self.categories.append(self.categorie(label))
                            # self.label.append(label[1])
                        InfoIndex = -1
                        for index,element in enumerate(self.kLabelInfo):
                            # print(element[0])
                            # print(label[0])
                            if element[0] == label[0]:
                                InfoIndex = index
                                break
                        points=shapes['points']
                        self.annotations.append(self.annotation(points,label,self.image(data,num),InfoIndex))
                        self.annID+=1

    def image(self,data,num):
        image={}
        img = utils.img_b64_to_arr(data['imageData'])  # 解析原图片数据
        # img=io.imread(data['imagePath']) # 通过图片路径打开图片
        # img = cv2.imread(data['imagePath'], 0)
        height, width = img.shape[:2]
        img = None
        image['height']=height
        image['width'] = width
        # image['id']=num+1
        image['file_name'] = data['imagePath'].split('\\')[-1]
        image['id'] = int((image['file_name'].split('.'))[0])
        self.height=height
        self.width=width
        return image

    def categorie(self,label):
        categorie={}
        # categorie['supercategory'] = label[0]
        # categorie['id']=len(self.label)+1 # 0 默认为背景
        # categorie['name'] = label[1]
        categorie['supercategory'] = 'person'
        categorie['id'] = 1  # 0 默认为背景
        categorie['name'] = 'person'
        categorie['keypoints'] = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                                  "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                                  "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
        categorie['skeleton'] = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                                 [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        return categorie

    def annotation(self,points,label,images,InfoIndex):
        annotation={}
        # print(list(np.asarray(points).flatten()))
        masks =list(map(float,np.asarray(points).flatten()))
        # print(masks)
        annotation['segmentation']=[masks]
        annotation['iscrowd'] = 0
        annotation['image_id'] = images['id']
        # # annotation['bbox'] = str(self.getbbox(points)) # 使用list保存json文件时报错（不知道为什么）
        # # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] 使用该方式转成list
        annotation['area'] = self.ComputePolygonArea(list(np.asarray(points).flatten()))
        # # print(annotation['area'])
        # print(self.getbbox(points))
        annotation['bbox'] = list(map(float,self.getbbox(points)))
        if(InfoIndex == -1):
            annotation['num_keypoints'] = 0
            annotation['keypoints'] = []
        else:
            annotation['num_keypoints'] = self.kLabelInfo[InfoIndex][2]
            # print(self.kLabelInfo[InfoIndex][1])
            annotation['keypoints'] = self.kLabelInfo[InfoIndex][1]
        # print(annotation['keypoints'])
        # annotation['category_id'] = self.getcatid(label)
        annotation['category_id'] = 1
        annotation['id'] = self.annID
        return annotation

    def getcatid(self,label):
        for categorie in self.categories:
            if label[1]==categorie['name']:
                return categorie['id']
        return -1

    def ComputePolygonArea(self,point2D):
        pointNum = point2D.__len__()
        points = []
        for i in range(0, pointNum, 2):
            points.append([point2D[i], point2D[i + 1]])
        pointsNum = len(points)
        s = points[0][1] * (points[pointsNum - 1][0] - points[1][0])
        for i in range(1, pointsNum):
            s += points[i][1] * (points[i - 1][0] - points[(i + 1) % pointsNum][0])
        return np.fabs(s / 2.0)

    def getbbox(self,points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points
        mask = self.polygons_to_mask([self.height,self.width], polygons)
        # print(mask)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c-left_top_c, right_bottom_r-left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self,img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco={}
        data_coco['images']=self.images
        data_coco['categories']=self.categories
        data_coco['annotations']=self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'),indent=4)  # indent=4 更加美观显示

labelme_json=glob.glob('/home/myubuntu/Desktop/姿态估计标注0张-30张/*.json')
# labelme_json=['./1.json']

labelme2coco(labelme_json,'/home/myubuntu/Desktop/姿态估计标注0张-30张/train_val_new.json')
