from lightning import LightningDataModule
import os
from PIL import Image
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, ToTensor


class GRIDEYEDataset(Dataset):
    def __init__(self, root, annotation=None, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = None
        self.ids = []
        self.categories = []

        if annotation is not None:
            self.coco = COCO(annotation)
            self.ids = list(sorted(self.coco.imgs.keys()))
            self.categories = self.coco.loadCats(self.coco.getCatIds())

    def __getitem__(self, index):
        img_id = self.ids[index] if self.coco else index

        if self.coco:
            # Gets all annotation ids for given image id
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            coco_annotation = self.coco.loadAnns(ann_ids)
            path = self.coco.loadImgs(img_id)[0]['file_name']

            # Open current image in the dataloader
            path = os.path.join('images', path)
            img = Image.open(os.path.join(self.root, path))

            # Get number of objects in image
            num_objs = len(coco_annotation)

            boxes = []
            category_ids = []
            categories = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]

            # For each object, get its bounding box
            for i in range(num_objs):
                xmin = coco_annotation[i]["bbox"][0]
                ymin = coco_annotation[i]["bbox"][1]
                xmax = xmin + coco_annotation[i]["bbox"][2]
                ymax = ymin + coco_annotation[i]["bbox"][3]
                boxes.append([xmin, ymin, xmax, ymax])
                category_id = coco_annotation[i]['category_id']
                category_ids.append(categories.index(category_id))
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(category_ids, dtype=torch.int64)

            img_id = torch.tensor([img_id])
            areas = []
            for i in range(num_objs):
                areas.append(coco_annotation[i]["area"])
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            my_annotation = {
                'boxes': boxes,
                'labels': labels,
                'image_id': img_id,
                'area': areas,
                'iscrowd': iscrowd
            }
        else:
            path = os.path.join('images', str(img_id) + '.png')
            img = Image.open(os.path.join(self.root, path))
            my_annotation = {}

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids) if self.ids else len(os.listdir(self.root))


def get_transform():
    custom_transforms = []
    custom_transforms.append(ToTensor())
    return Compose(custom_transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


class DataModule(LightningDataModule):
    def __init__(self, data_dir):
        super().__init__()
        self.batch_size = 10
        self.data_dir = data_dir
        self.image_dir = os.path.join(self.data_dir, 'images')
        self.transforms = get_transform()
        self.dataloader_params = {
            'batch_size': 1,
            'shuffle': True,
            'num_workers': 2,
            'collate_fn': collate_fn
        }
        self.filename = 'annotations.json'

    def setup(self, stage):
        if stage == 'fit':
            train_coco = os.path.join(self.data_dir, self.filename)
            dataset = GRIDEYEDataset(root=self.data_dir, annotation=train_coco, transforms=self.transforms)
            train_set_size = int(len(dataset) * 0.8)
            valid_set_size = len(dataset) - train_set_size
            self.train_dataset, self.valid_dataset = random_split(dataset, [train_set_size, valid_set_size])

        if stage == 'test':
            test_coco = os.path.join(self.data_dir, self.filename)
            self.test_dataset = GRIDEYEDataset(root=self.data_dir, annotation=test_coco, transforms=self.transforms)

        if stage == 'predict':
            self.predict_dataset = GRIDEYEDataset(root=self.data_dir, transforms=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.dataloader_params)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, **self.dataloader_params)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.dataloader_params)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, **self.dataloader_params)