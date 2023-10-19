from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from utils import pred_lines
import os
import argparse


parser = argparse.ArgumentParser('M-LSD demo')
parser.add_argument('--model_path', default='tflite_models/M-LSD_512_large_fp32.tflite', type=str, help='path to tflite model')
parser.add_argument('--input_size', default=512, type=int, choices=[512, 320], help='input size')
args = parser.parse_args()

# Load tflite model
interpreter = tf.lite.Interpreter(model_path=args.model_path)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def gradio_wrapper_for_LSD(img_input, score_thr, dist_thr):

    lines = pred_lines(img_input, interpreter, input_details, output_details,
                       input_shape=[args.input_size, args.input_size], score_thr=score_thr, dist_thr=dist_thr)

    img_output = img_input.copy()

    # draw lines
    for line in lines:
        x_start, y_start, x_end, y_end = [int(val) for val in line]
        cv2.line(img_output, (x_start, y_start), (x_end, y_end), [0, 255, 255], 2)

    return img_output
def main(args):

    img = cv2.imread(args.image_list)
    lines = gradio_wrapper_for_LSD(img, args.score, args.dist)


    out = args.outputs
    os.makedirs(out, exist_ok=True)
    cv2.imwrite(f'{out}2_1_out.jpg', lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('M-LSD demo')
    parser.add_argument('--model_path', default='tflite_models/M-LSD_512_large_fp32.tflite', type=str,
                        help='path to tflite model')
    parser.add_argument('--input_size', default=512, type=int, choices=[512, 512], help='input size')
    parser.add_argument('--outputs', default='outputs/', type=str, help='outputs')
    parser.add_argument('--image_list', default='./2_1.jpg', type=str, help='input size')
    parser.add_argument('--score', default=0.3, type=int, help='input size')
    parser.add_argument('--dist', default=4, type=int, help='input size')
    args = parser.parse_args()
    main(args)