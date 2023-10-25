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
folder_path = './book/train2017/'
img_dir = './book/images/1.jpg'
# 初始化结果列表
result = []
result1 = []
a = 1
# 遍历文件夹中的每个文件
for filename in os.listdir(folder_path):
    name = filename.split('.')[0]+'.jpg'
    # 检查文件是否为json文件
    if filename.endswith('.txt'):
        # 打开并读取json文件
        new_data = {
            "filename": filename,
            "lines": [],
            "height": 1440,
            "width": 2560
        }
        with open(os.path.join(folder_path, filename), 'r') as f:
            data = f.readlines()
            # 图像的尺寸
            width = 2560
            height = 1440

            # 将归一化的坐标还原为原始的像素坐标
            for line in data:
                # 提取所需的信息
                normalized_coords = [float(i) for i in line.split(' ')[1:]]
                if len(normalized_coords) == 8:
                    contents = []
                    for i in range(0, len(normalized_coords), 2):
                        x = normalized_coords[i] * width
                        y = normalized_coords[i + 1] * height
                        contents.append(x)
                        contents.append(y)
                    # 遍历shapes并提取points
                    mask = np.zeros([height,width],dtype=np.uint8)
                    new_points1 = [contents[0], contents[1],contents[2], contents[3]]
                    new_points2 = [contents[0], contents[1],contents[6], contents[7]]
                    new_points3 = [contents[2], contents[3], contents[4], contents[5]]
                    new_points4 = [contents[6], contents[7], contents[4], contents[5]]

                    # cv2.line(image,[new_points1[0],new_points1[1]],[new_points1[2],new_points1[3]],[0, 255, 255], 2)
                    # cv2.line(image, [new_points2[0],new_points2[1]],[new_points2[2],new_points2[3]], [0, 255, 255], 2)
                    # cv2.line(image, [new_points3[0],new_points3[1]],[new_points3[2],new_points3[3]], [0, 255, 255], 2)
                    # cv2.line(image, [new_points4[0],new_points4[1]],[new_points4[2],new_points4[3]], [0, 255, 255], 2)

                    new_data["lines"].append(new_points1)
                    new_data["lines"].append(new_points2)
                    new_data["lines"].append(new_points3)
                    new_data["lines"].append(new_points4)
                # 将新的数据添加到结果列表中
        a +=1
        if a<= 400:
            result.append(new_data)
        else:
            result1.append(new_data)

#将结果保存为新的json文件
with open('book/train.json', 'w') as f:
    json.dump(result, f)
with open('book/valid.json', 'w') as f:
    json.dump(result1, f)

