import tensorflow as tf
import numpy as np
import cv2

global model
saved_model='./snapshots/model.h5'
model = tf.keras.models.load_model(saved_model)

def inference(
    img_path,
    stage_num,
    keypoints_num,
    inputs_width,
    inputs_height,
    img_width,
    img_height,
    score_map_w,
    score_map_h):
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (inputs_width, inputs_height)) / 255.0
    img = img.astype('float32')
    img = np.expand_dims(img, 0)

    pred = model.predict(img)

    points = []

    for i in range(keypoints_num):
        max_index = np.where(
            pred[stage_num-1][0][:,:,i] == np.max(pred[stage_num-1][0][:,:,i]))

        points.append(
            [max_index[1][0] * (img_width / score_map_w), 
            max_index[0][0] * (img_height / score_map_h)])
    
    return points

points = inference(
    img_path='C:/Users/yuanmingqi/Desktop/cow_gait/images/00010100000006E8-20190906-165253-936.jpg',
    stage_num=6,
    keypoints_num=10,
    inputs_width=368,
    inputs_height=368,
    img_width=1280,
    img_height=720,
    score_map_w=46,
    score_map_h=46)

import matplotlib.pyplot as plt
img_path = 'C:/Users/yuanmingqi/Desktop/cow_gait/images/00010100000006E8-20190906-165253-936.jpg'
for item in points:
    img = cv2.imread(img_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for item in points:
        cv2.circle(img, tuple(np.int32(np.array(item))), radius=5, color = (0, 255, 0), thickness = -1)
plt.figure(figsize=(16, 16), facecolor='w')
plt.imshow(img)
plt.show()