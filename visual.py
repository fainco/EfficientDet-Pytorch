# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time

import torch
from torch.backends import cudnn
import os
from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

def display(preds, imgs, file_index,imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (255, 255, 0), 2)

        if imshow:
            cv2.namedWindow("img",0)
            cv2.resizeWindow("img", 1024, 1024)
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite(f'test/defect1/{file_index}.jpg', imgs[i])

# obj_list = ['TPFP0','TTPIG','TTFBG','TPPP5','TPDPD']
obj_list = ['DEFECT']
train_data_path = './datasets/coco/val2017'
train_file_name = os.listdir(train_data_path)
imgs_path = [os.path.join(train_data_path,img_name) for img_name in train_file_name]

compound_coef = 0
model_weight = 'efficientdet-d0_classnum_1_2.pth'
force_input_size = None  # set None to use default size
# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
# model.load_state_dict(torch.load(f'weights/{model_weight}'))
model.load_state_dict(torch.load(f'./logs/coco/efficientdet-d0_117_15000.pth'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()


# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
####

for img_path in imgs_path:
    # time.sleep(1)
    t1 = time.time()

    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)
    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

    out = invert_affine(framed_metas, out)
    t2 = time.time()
    tact_time = (t2 - t1) / 10
    # print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

    display(out, ori_imgs, file_index = img_path.split('/')[-1].split('.')[0],imshow=True, imwrite=False)
