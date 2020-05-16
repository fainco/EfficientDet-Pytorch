import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2


class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None):

        self.root_dir = root_dir # datasets/coco/
                                 # datasets/coco/ : 这个目录下有annotations、train2017、test2017、val2017;
                                 # annotations ：这个目录下有instances_train2017、instances_test2017、instances_val2017；
        self.set_name = set
        self.transform = transform #数据预处理的方法集合。

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds() #得到所有图片在标注文件中的id.方便后期训练读取

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        # 将所有类别读取出来，并排序。
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {} # {name:cls_len_index} 为每个类别建立索引
        self.coco_labels = {} # {0:0,1:1,...}({cls_len_index:cate_id})
        self.coco_labels_inverse = {} # {0:0,1:1,...}({cate_id:cls_len_index})
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {} # {cls_len_index:name} 方便以后根据预测序号找到实际类别名称
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0] #通过image的id找到对应的image_info, 
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])# image_info中包含了image的file_name
        img = cv2.imread(path)                                         # 根据file_name结合路径，就可以做图片读取操作
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转换图片的通道

        return img.astype(np.float32) / 255.  # 图像缩小到【0,1】范围内

    def load_annotations(self, image_index):
        # get ground truth annotations
        # 根据图片的索引得到该图片所有标注信息的id，
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5)) # 当没有标注信息的时候，直接返回该变量box:4+category_id:1

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations # 根据标注id找到标注的详细信息。
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            # 对于没有宽或者高的标注，直接跳过
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    # 这批图片中最多annot是多少，之后会对不足这个数值的进行-1值pad.
    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    """
        1. 等比例缩放长宽，缩放后长宽比不变；
        2. padding长宽到指定img_size;
        3. 按照缩放比例修改annots；
    """
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        ## padding操作，将图像长宽padding到512大小。
        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image
        
        ## 因为缩放了，所以修改标注宽的数值
        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""
    """
        1. 数据增强，将图片按照第二维度反转；
        2. 修改annots;
    """
    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy() # copy is deepcopy in numpy
            x2 = annots[:, 2]#.copy()

            # x_tmp = x1.deepcopy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x1

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
