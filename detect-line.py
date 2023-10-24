from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from utils import pred_lines
import os
import argparse
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser('M-LSD demo')
parser.add_argument('--model_path', default='tflite_models/M-LSD_512_large_fp32.tflite', type=str, help='path to tflite model')
parser.add_argument('--input_size', default=512, type=int, choices=[512, 320], help='input size')
args = parser.parse_args()

# Load tflite model
interpreter = tf.lite.Interpreter(model_path=args.model_path)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def length(self):
        return ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2) ** 0.5

    def slope(self):
        return (self.y2 - self.y1) / (self.x2 - self.x1)

    def intercept(self):
        return self.y1 - self.slope() * self.x1


def gradio_wrapper_for_LSD(img_input, score_thr, dist_thr):

    lines = pred_lines(img_input, interpreter, input_details, output_details,
                       input_shape=[args.input_size, args.input_size], score_thr=score_thr, dist_thr=dist_thr)

    img_output = img_input.copy()

    # draw lines
    for line in lines:
        x_start, y_start, x_end, y_end = [int(val) for val in line]
        cv2.line(img_output, (x_start, y_start), (x_end, y_end), [0, 255, 255], 2)

    return img_output, lines

def slope_cluster(slopes):
    slopes_array = np.array(slopes).reshape(-1, 1)
    # 使用K-means聚类算法对斜率进行聚类
    kmeans = KMeans(n_clusters=2)  # 假设我们想要将线段分为3组
    kmeans.fit(slopes_array)
    # 输出每个线段的标签（即它们所属的组）
    labels = kmeans.labels_
    print(labels)
def dot_product(line1, line2):
    return (line1.x2 - line1.x1) * (line2.x2 - line2.x1) + (line1.y2 - line1.y1) * (line2.y2 - line2.y1)

def projection_length(line, base_line):
    return dot_product(line, base_line) / base_line.length()

def calculate(p,a,b):

    # 将元组转换为NumPy数组
    p = np.array(p)
    a = np.array(a)
    b = np.array(b)

    # 计算投影点
    ap = p - a
    ab = b - a
    result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab

    return tuple(result)
def GeneralEquation(first_x,first_y,second_x,second_y):
    A = second_y - first_y
    B = first_x - second_x
    C = second_x * first_y - first_x * second_y
    return A, B, C

def GetIntersectPointofLines(x1,y1,x2,y2,x3,y3,x4,y4):
    # from http://www.cnblogs.com/DHUtoBUAA/
    A1,B1,C1=GeneralEquation(x1,y1,x2,y2)
    A2, B2, C2 = GeneralEquation(x3,y3,x4,y4)
    m=A1*B2-A2*B1
    if m==0:
        print("无交点")
    else:
        x=(C2*B1-C1*B2)/m
        y=(C1*A2-C2*A1)/m

    return int(x), int(y)
def main(args):

    out = args.outputs
    img = cv2.imread(args.image_list)
    img1 = img.copy()
    h, w, _ =img.shape
    name = os.path.basename(args.image_list).split('.')[0]
    images, lines = gradio_wrapper_for_LSD(img, args.score, args.dist)
    lines_ = []
    for line in lines:
        lines_.append(Line(*line))

    # max_coordinates = ((longest_line.x1), (longest_line.y1)), ((longest_line.x2), (longest_line.y2))


    slopes = []
    slope = []
    lengths = []
    length = []
    cluster = []
    avg_slope = []

    #平均斜率
    for line in lines_:
        if line.length() < 40 and line.slope() > 0 and line.slope()  <= 2:
            avg_slope.append(line.slope())

    avg_slopes = np.mean(avg_slope)
    for len in lines_:
        lengths.append(len.length())
    #坐标及斜率
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask1 = np.zeros(img.shape, dtype=np.uint8)
    #左侧
    pt1 = (0, int(avg_slopes * 0 +h/2))
    pt2 = (w, int(avg_slopes * w +h/2))
    a = 1
    # baseline
    cv2.line(img, pt1, pt2, (255, 255, 0), 2)
    cv2.line(mask1, pt1, pt2, (255, 255, 0), 2)
    for line in lines_:
        x1 = int(line.x1)
        y1 = int(line.y1)
        x2 = int(line.x2)
        y2 = int(line.y2)

        #base
        #投影
        x_r =calculate(p=(x1, y1), a=pt1, b=pt2)
        y_r =calculate(p=(x2, y2), a=pt1, b=pt2)

        x_r = (int(round(x_r[0])), int(round(x_r[1])))
        y_r = (int(round(y_r[0])), int(round(y_r[1])))
        slopes.append(line.slope())
        #if line.length() < 100 and abs(line.slope()) < 2:
        if line.length() < 50 and line.slope() > 0 and line.slope() <= 1.5:

            cv2.circle(img, pt1,10,(0,0,255),-1)
            cv2.circle(img, pt2, 10, (0, 0, 255), -1)

            cv2.line(img, (x1, y1), (x2, y2), [255, 0, 0], 2)
            cv2.line(mask, (x1, y1), (x2, y2), [255, 0, 0], 2)

            #t投影
            cv2.line(mask, x_r, y_r, [0,0,255], 2)
            cv2.line(img,  x_r, y_r, [0,0,255], 2)

            slope.append(line.slope())
            length.append(line.length())
            cluster.append((line.slope()))

            a +=1

        #竖线
        elif line.slope() >2:
            # projection_length = int(calculate(x1, y1, x2, y2))
            cv2.line(img1, (x1, y1), (x2, y2), [0, 255, 255], 2)

            x1, y1 = GetIntersectPointofLines(x1, y1, x2, y2, 0, h, w, h)
            x2, y2 = GetIntersectPointofLines(x1, y1, x2, y2, 0, 0, 2, 0)
            cv2.line(mask, (x1, y1), (x2, y2), [0, 255, 0], 2)
            cv2.line(img, (x1, y1), (x2, y2), [0, 255, 0], 2)


    # cv2.imwrite(f'{out}{name}_mask.jpg', mask)
    #散点
    # slope_cluster(cluster)
    # plt.scatter(length, slope)
    # plt.title("Scatter Plot of Lengths vs Slopes")
    # plt.xlabel("Length")
    # plt.ylabel("Slope")
    # plt.show()

    # plt.figure()
    # plt.subplot(131)
    # plt.imshow(mask)
    # plt.subplot(132)
    # plt.imshow(img)
    # plt.subplot(133)
    # plt.imshow(mask1)
    # plt.show()

    # plt.savefig(f'{out}{name}_out.jpg')
    # 创建散点图

    # #斜率聚类
    # slope_cluster(slopes)

    out = os.path.join(out, name)
    os.makedirs(out, exist_ok=True)
    cv2.imwrite(f'{out}/{name}_out.jpg', images)
    cv2.imwrite(f'{out}/{name}_mask.jpg', mask)
    cv2.imwrite(f'{out}/{name}_ori.jpg', img)
    cv2.imwrite(f'{out}/{name}_col.jpg', img1)
if __name__ == '__main__':
    parser = argparse.ArgumentParser('M-LSD demo')
    parser.add_argument('--model_path', default='tflite_models/M-LSD_512_large_fp32.tflite', type=str,
                        help='path to tflite model')
    parser.add_argument('--input_size', default=512, type=int, choices=[512, 512], help='input size')
    parser.add_argument('--outputs', default='outputs/', type=str, help='outputs')
    parser.add_argument('--image_list', default='./assets/gt_1_2.jpg', type=str, help='input size')
    parser.add_argument('--score', default=0.15, type=int, help='input size')
    parser.add_argument('--dist', default=3, type=int, help='input size')
    args = parser.parse_args()
    main(args)