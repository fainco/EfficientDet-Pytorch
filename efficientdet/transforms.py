import os
import torch
import numpy as np
import cv2
import random

def getMaxThread(image):
    """
     # 如果没有归一化到0，1则像素最大值设为255
    """
    if image.max() > 10:
        maxthread = 255
    else:
        maxthread = 1
    return maxthread


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        
    """

    def __init__(self ):
        pass

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        h,w,c = image.shape
        
        annots[:,2] = annots[:,2] + annots[:,0]
        annots[:,3] = annots[:,3] + annots[:,1]
        # 将目标最大边界找出
        xmin = max(annots[:,0].min()-10,0)
        ymin = max(annots[:,1].min()-10,0)
        xmax = min(annots[:,2].max()+10,w)
        ymax = min(annots[:,3].max()+10,h)
        
        top = np.random.randint(0,int(ymin//2))
        bottom = np.random.randint(int((h-ymax)//2+ymax),h)
        left = np.random.randint(0,int(xmin//2))
        right = np.random.randint(int((w-xmax)//2+xmax),w)
        
        # 使得长宽相同，之后缩放不会拉伸
        max_size = max(bottom - top,right - left)
        bottom = top + max_size
        right = left + max_size
        
        print([top,bottom,left,right])
        image = image[top: bottom,left: right]

        annots[:,2] = annots[:,2] - annots[:,0]
        annots[:,3] = annots[:,3] - annots[:,1]
        print(annots)
        annots[:,:2] = annots[:,:2] - [left,top]
        print(annots)
        return {'img': image, 'annot': annots}


class RandomFlip(object):
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        if random.random()<0.5:
            image = cv2.flip(image,1)
            annots[:,0] = image.shape[1]-annots[:,0] - annots[:,2]
        if random.random()<0.5:
            image = cv2.flip(image,0)
            annots[:,1] = image.shape[1]-annots[:,1] - annots[:,3]

        return {'img': image, 'annot': annots}


class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']
        if random.random() < self.p:
            image = cv2.GaussianBlur(image,(15,15),0)
        return {'img': image, 'annot': annots}


class RandomSwap(object):
    """
    # 随机变换通道
    """
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        perms = ((0, 1, 2), (0, 2, 1),
                    (1, 0, 2), (1, 2, 0),
                    (2, 0, 1), (2, 1, 0))
        if random.random() < 0.5:
            swap = perms[random.randrange(0, len(perms))]
            image = image[:, :, swap]
        return {'img': image, 'annot': annots}


class RandomContrast(object):
    """
    # 随机变换对比度
    """
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
    def __call__(self, sample):    
        image, annots = sample['img'], sample['annot']
        if random.random() < 0.5:
            # image = image.astype(np.float32).copy()
            maxthread = getMaxThread(image)
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
            image = image.clip(min=0, max=maxthread)
        return {'img': image, 'annot': annots}


class RandomSaturation(object):
    """
    # 随机变换饱和度
    """
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
    def __call__(self, sample):    
        image, annots = sample['img'], sample['annot']    
        if random.random() < 1.5:
            # image = image.astype(np.float32).copy()
            rotate = random.uniform(self.lower, self.upper)
            print(rotate)
            c = random.randint(0,2)
            maxthread = getMaxThread(image)
            image[:, :, c] = np.clip(image[:, :, c] * rotate,0,maxthread)
        return {'img': image, 'annot': annots}


### not use
class RandomHue(object):

    def __init__(self, delta=90/255):
        self.delta = delta

    # 随机变换色度(HSV空间下(-180, 180))
    def __call__(self, sample):    
        image, annots = sample['img'], sample['annot']    
        if random.random() < 0.5:
            maxthread = getMaxThread(image)
            self.delta *= maxthread
            # image = image.astype(np.float32).copy()
            image[:, :, 0] =  np.clip(image[:, :, 0] + random.uniform(-self.delta, self.delta),0,maxthread)
        return {'img': image, 'annot': annots} 