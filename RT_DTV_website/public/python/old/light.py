from collections import defaultdict
#import tensorflow as tf
import cv2
import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import torch
import shutil
from screenshot import make_violation_image




def light_predict(model, light_info, output_folder, filename, save = 1):
    result_path = os.path.join(output_folder, 'result', filename)
    light_info_folder = os.path.join(output_folder, 'light_info')
    carimg_folder = os.path.join(output_folder, 'carimg')
    result = []

    for id, frame_info in light_info.items():
        violation_imgs = []
        violation_bboxes = []
        light_box_num_list = []
        frame_num = len(frame_info)
        count = 0
        for _, info in frame_info.items():
            car_img = info['car_imgs']
            if count == int(0.4 * frame_num) or count == int(0.6 * frame_num) or count == int(0.8 * frame_num):
                violation_imgs.append(info['ori_imgs'])
                violation_bboxes.append(info['bboxes'])
            count = count + 1
            pred = model.predict(car_img, save=False,verbose=False)
            if len(pred[0].boxes) == 0:
                light_box_num_list.append(len(pred[0].boxes))
            else:
                light_box_num_list.append(1)
        light = is_light(light_box_num_list)
        if light[0] == 0:
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            license_plate = 'ccc-0001'
            make_violation_image(violation_imgs, violation_bboxes, id, license_plate, result_path)
            result.append(id)
    #print(carID, light)
    #畫波型圖
    if save:
        plt.plot(light_box_num_list)
        plt.title(f"light Information")
        plt.xlabel('Index')
        plt.ylabel('light')

        # plt.savefig(output_path)
        plt.cla()  
    return result  

#判斷打燈(0:沒有打燈 1:有打燈)
def is_light(num_list):
    upper = 0
    lower = 0
    wave_num = 0
    frame = len(num_list)
    flag = 0
    flag2 = 0
    flag3 = 0
    #改成除以第一個出現1的地方(前面車直行不計)
    for i in num_list:
        if i == 1:
            upper += 1
            flag = 1
        elif i == 0 and flag == 1:
            lower += 1
    for i in num_list:
        if i == 1 and flag2 == 0:
            flag2 = 1
        elif i == 1 and flag2 == 1 and flag3 == 0:
            wave_num += 1
            flag3 = 1
        elif i == 0:
            flag2 = 0
            flag3 = 0
    #錯誤修正(如果是一直線會除以0，所以要避免)
    if upper == 0 and lower == 0:
        lower = 1

    if upper / (upper + lower) < 0.5 and wave_num/frame < 0.04:
        light = 0 
    else:
        light = 1

    return [light, upper / (upper + lower), wave_num/frame]









    










