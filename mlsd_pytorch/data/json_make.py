import json
import tqdm
import os
import cv2
import torch
import json
import random
import numpy as np

import matplotlib.pyplot as plt

#检查是否四个一组
# with open('./wireframe/valid.json', 'r') as f:
#     content = json.load(f)
#     for item in content:
#         lines = item['lines']
#         img = item['filename']
#
#         for line in lines:
#             if len(line) != 4:
#                 print(f"在文件{item['filename']}")

# 文件夹路径
folder_path = './test/'
img_dir = './book/images/1.jpg'
# 初始化结果列表
result = []
result1 = []
# 遍历文件夹中的每个文件
file = os.listdir(folder_path)
random.shuffle(file)
for index, filename in enumerate(file):
    name = filename.split('.')[0]+'.jpg'
    path = os.path.join(folder_path, filename)
    # 检查文件是否为json文件
    if filename.endswith('.json'):
        with open(path,'r') as f:
            content = json.load(f)
            new_data = {
                    "filename": filename,
                    "lines": [],
                    "height": content["imageHeight"],
                    "width": content["imageWidth"]
                }
            # 打开并读取json文件
            data = content['shapes']
            for i in data:
                i = i['points']
                mask = np.zeros([content["imageHeight"], content["imageWidth"]], dtype=np.uint8)
                if len(i) ==4:
                    new_points1 = [i[0][0], i[0][1], i[1][0], i[1][1]]
                    new_points2 = [i[0][0], i[0][1], i[3][0], i[3][1]]
                    new_points3 = [i[2][0], i[2][1], i[1][0], i[1][1]]
                    new_points4 = [i[2][0], i[2][1], i[3][0], i[3][1]]
                    new_data["lines"].append(new_points1)
                    new_data["lines"].append(new_points2)
                    new_data["lines"].append(new_points3)
                    new_data["lines"].append(new_points4)
                else:
                    new_points_lien = [i[0][0], i[0][1], i[1][0], i[1][1]]
                    new_data["lines"].append(new_points_lien)
    if index <= len(file)*0.8:
        result.append(new_data)
    else:
        result1.append(new_data)
#将结果保存为新的json文件
with open('book/train.json', 'w') as f:
    json.dump(result, f)
with open('book/valid.json', 'w') as f:
    json.dump(result1, f)

