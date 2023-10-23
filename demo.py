import torch
import os
import sys
import cv2

from models.mbv2_mlsd_tiny import  MobileV2_MLSD_Tiny
from models.mbv2_mlsd_large import  MobileV2_MLSD_Large

from utils import  pred_lines



def main():
    current_dir = os.path.dirname(__file__)
    if current_dir == "":
        current_dir = "./"
    # model_path = current_dir+'/models/mlsd_tiny_512_fp32.pth'
    # model = MobileV2_MLSD_Tiny().cuda().eval()

    model_path = current_dir + '/models/mlsd_large_512_fp32.pth'
    model = MobileV2_MLSD_Large().cuda().eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

    img_fn = current_dir+'/assets/2_3.jpg'

    img = cv2.imread(img_fn)
    h, w, _ = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lines = pred_lines(img, model, [512,512], 0.15, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for l in lines:
        cv2.line(img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0,255,255), 2,16)
    cv2.imwrite(current_dir+'/data/2_1_out.jpg', img)

if __name__ == '__main__':
    main()
