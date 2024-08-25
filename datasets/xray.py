from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv
import os
import torch
import random
import numpy as np
from PIL import Image
from .tokenizers import Tokenizer
from .utils import nested_tensor_from_tensor_list, read_json


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


def get_transform(MAX_DIM):
    def under_max(image):
        if image.mode != 'RGB':
            image = image.convert("RGB")

        shape = np.array(image.size, dtype=np.float)
        long_dim = max(shape)
        scale = MAX_DIM / long_dim

        new_shape = (shape * scale).astype(int)
        image = image.resize(new_shape)

        return image

    train_transform = tv.transforms.Compose([
        RandomRotation(),
        tv.transforms.Lambda(under_max),
        tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
            0.8, 1.5], saturation=[0.2, 1.5]),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    val_transform = tv.transforms.Compose([
        tv.transforms.Lambda(under_max),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return train_transform, val_transform


transform_class = tv.transforms.Compose([
    tv.transforms.Resize(224),
    tv.transforms.CenterCrop((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class XrayDataset(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=None, transform_class=transform_class,
                 mode='training', data_dir=None, dataset_name=None, image_size=None,
                 theta=None, gamma=None, beta=None):
        super().__init__()

        self.root = root
        self.transform = transform
        self.transform_class = transform_class
        self.annot = ann

        self.data_dir = data_dir
        self.image_size = image_size

        self.theta = theta
        self.gamma = gamma
        self.beta = beta

        if mode == 'training':
            self.annot = self.annot[:]
        else:
            self.annot = self.annot[:]
        if dataset_name == "mimic_cxr":
            threshold = 10
        elif dataset_name == "iu_xray":
            threshold = 3
        self.data_name = dataset_name
        self.tokenizer = Tokenizer(ann_path=root, threshold=threshold, dataset_name=dataset_name)
        self.max_length = max_length + 1

    def _process(self, image_id):
        val = str(image_id).zfill(12)
        return val + '.jpg'

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        caption = self.annot[idx]["report"]
        image_path = self.annot[idx]['image_path']
        image = Image.open(os.path.join(self.data_dir, image_path[0])).resize((300, 300)).convert('RGB')
        class_image = image
        com_image = image

        if self.data_name == "mimic_cxr":
            mask_arr = np.load(os.path.join(self.data_dir.strip("images300"), "images300_array",
                                            image_path[0].replace(".jpg", ".npy")))
        else:
            mask_arr = np.load(os.path.join(self.data_dir.strip("images"), "images300_array",
                                            image_path[0].replace(".png", ".npy")))

        if (np.sum(mask_arr) / 90000) > self.theta:
            image_arr = np.asarray(image)
            boost_arr = image_arr * np.expand_dims(mask_arr, 2)
            weak_arr = image_arr * np.expand_dims(1 - mask_arr, 2)
            image = Image.fromarray(boost_arr + (weak_arr * self.gamma).astype(np.uint8))

        if self.transform:
            image = self.transform(image)
            com_image = self.transform(com_image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0), max_dim=self.image_size)
        com_image = nested_tensor_from_tensor_list(com_image.unsqueeze(0), max_dim=self.image_size)

        if self.transform_class:
            class_image = self.transform_class(class_image)

        caption = self.tokenizer(caption)[:self.max_length]
        cap_mask = [1] * len(caption)
        return image.tensors.squeeze(0), image.mask.squeeze(0), com_image.tensors.squeeze(0), com_image.mask.squeeze(
            0), caption, cap_mask, class_image

    @staticmethod
    def collate_fn(data):
        max_length = 129
        image_batch, image_mask_batch, com_image_batch, com_image_mask_batch, report_ids_batch, report_masks_batch, class_image_batch = zip(
            *data)
        image_batch = torch.stack(image_batch, 0)
        image_mask_batch = torch.stack(image_mask_batch, 0)
        com_image_batch = torch.stack(com_image_batch, 0)
        com_image_mask_batch = torch.stack(com_image_mask_batch, 0)
        class_image_batch = torch.stack(class_image_batch, 0)
        target_batch = np.zeros((len(report_ids_batch), max_length), dtype=int)
        target_masks_batch = np.zeros((len(report_ids_batch), max_length), dtype=int)

        for i, report_ids in enumerate(report_ids_batch):
            target_batch[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(report_masks_batch):
            target_masks_batch[i, :len(report_masks)] = report_masks
        target_masks_batch = 1 - target_masks_batch

        return image_batch, image_mask_batch, com_image_batch, com_image_mask_batch, torch.tensor(
            target_batch), torch.tensor(target_masks_batch, dtype=torch.bool), class_image_batch


def build_dataset(config, mode='training', anno_path=None, data_dir=None, dataset_name=None, image_size=None,
                  theta=None, gamma=None, beta=None):
    train_transform, val_transform = get_transform(MAX_DIM=image_size)
    if mode == 'training':
        train_file = anno_path
        data = XrayDataset(train_file, read_json(
            train_file)["train"], max_length=config.max_position_embeddings, limit=config.limit,
                           transform=train_transform,
                           mode='training', data_dir=data_dir, dataset_name=dataset_name, image_size=image_size,
                           theta=theta, gamma=gamma, beta=beta)
        return data

    elif mode == 'validation':
        val_file = anno_path
        data = XrayDataset(val_file, read_json(
            val_file)["val"], max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform,
                           mode='validation', data_dir=data_dir, dataset_name=dataset_name, image_size=image_size,
                           theta=theta, gamma=gamma, beta=beta)
        return data
    elif mode == 'test':
        test_file = anno_path
        data = XrayDataset(test_file, read_json(
            test_file)["test"], max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform,
                           mode='test', data_dir=data_dir, dataset_name=dataset_name, image_size=image_size,
                           theta=theta, gamma=gamma, beta=beta)
        return data
    else:
        raise NotImplementedError(f"{mode} not supported")
