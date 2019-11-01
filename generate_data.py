import numpy as np
import cv2
import glob
import json
import time
import os

def generate_data_from_labelme(
    inputs_width,  
    inputs_height, 
    img_width, 
    img_height,
    score_map_w,
    score_map_h, 
    image_path, 
    label_path, 
    save_dir):
    print("[INFO]: Creating training data!")
    print("[INFO]: Detected label file: %d" % (len(os.listdir(label_path))))
    print("[INFO]: Points list:")
    
    t_start = time.clock()

    labels = []

    for item in glob.glob(label_path + '*.json'):
        name = os.path.split(item)[1].split('.')[0] + '.jpg'
        file = open(item, 'r')
        labels.append({"image":name, "points":json.load(file)['shapes']})

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for item in labels:
        print("[INFO]: image---%s" % (item['image']))
        try:
            img = cv2.imread(image_path + item['image'], 1)
            img = cv2.resize(img, (inputs_width, inputs_height))
        except:
            raise FileNotFoundError("Plese recheck your image file path!")
        x_train.append(img)

        p_list = []
        for points in item['points']:
            print(points)
            p_list.append(generate_label(
                score_map_w, 
                score_map_h, 
                points['points'][0][0] * (score_map_w / img_width), 
                points['points'][0][1] * (score_map_h / img_height), 
                variance=2.0))
        
        y_train.append(p_list)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    y_train_temp = np.ones(shape=(y_train.shape[0], score_map_h, score_map_w, y_train.shape[1]))
    for i in range(y_train.shape[0]):
        for j in range(y_train.shape[1]):
            y_train_temp[i][:,:,j] = y_train[i][j,:,:]
    
    y_train = y_train_temp
    print(y_train.shape)
    np.savez(save_dir, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    t_end = time.clock()

    print("[INFO]: Training data generation accomplished!")
    print("[INFO]: Time cost %s s" % (t_end - t_start))

def generate_label(heatmap_width, heatmap_height, c_x, c_y, variance):
    gaussian_map = np.zeros((heatmap_height, heatmap_width))
    for x_p in range(heatmap_width):
        for y_p in range(heatmap_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map

generate_data_from_labelme(
    inputs_width=368,
    inputs_height=368,
    img_width=1280,
    img_height=720,
    score_map_w=46,
    score_map_h=46,
    image_path="C:/Users/yuanmingqi/Desktop/cow_gait/CPM/data/images/",
    label_path="C:/Users/yuanmingqi/Desktop/cow_gait/CPM/data/label/",
    save_dir="C:/Users/yuanmingqi/Desktop/cow_gait/CPM/data/train.npz")

import matplotlib.pyplot as plt
data = np.load('./data/train.npz')

plt.imshow(data['y_train'][0][:,:,0], cmap='summer')
plt.show()