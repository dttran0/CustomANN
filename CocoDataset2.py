import os
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
# MS COCO API
from pycocotools.coco import COCO

class CocoDataset(Dataset):
    def __init__(self, root_path, annotation_file, transforms=None):
        self.root = root_path
        self.transforms = transforms
        self.coco_annotation = COCO(annotation_file)
        #this retrieve the list of the id number (object) in MS COCO
        self.ids = list(sorted(self.coco_annotation.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    
    def __getitem__(self, idx):
        # Image ID
        img_id = self.ids[idx]
        # List: get annotation id from coco
        ann_ids = self.coco_annotation.getAnnIds(img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotate = self.coco_annotation.loadAnns(ann_ids)
        # path for input image
        path = self.coco_annotation.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        # number of objects in the image
        num_objs = len(coco_annotate)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels =[]
        areas = []
        for i in range(num_objs):
            xmin, ymin, width, height = coco_annotate[i]['bbox']
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(coco_annotate[i]['category_id'])
            areas.append(coco_annotate[i]['area'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation
