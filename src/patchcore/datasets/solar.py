import os
from enum import Enum

import PIL
import torch
from torchvision import transforms

# We won't be using classes, these are for the MVTec dataset
# _CLASSNAMES = [
#     "bottle",
#     "cable",
#     "capsule",
#     "carpet",
#     "grid",
#     "hazelnut",
#     "leather",
#     "metal_nut",
#     "pill",
#     "screw",
#     "tile",
#     "toothbrush",
#     "transistor",
#     "wood",
#     "zipper",
# ]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class SolarDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Solar.
    """

    def __init__(
        self,
        source,
        # classname,    # we won't be using classes, so commented out
        # resize=256,   # we won't be using resize value, so commented out
        image_width=1000,     # BE CAREFUL WITH CONSERVING ASPECT RATIO!
        image_height=224,     
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the Solar data folder.
            classname: [str or None]. Name of Solar class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            image_width: [int]. Width the loaded image gets
                         (center-)cropped to.
            image_height: [int]. Height the loaded image gets
                          (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize((image_height, image_width)),  # crop rectangular (H, W)
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        self.transform_img = transforms.Compose(self.transform_img)

        # comented out because we are not going to use ground truth masks
        # self.transform_mask = [
        #     transforms.Resize(resize),
        #     transforms.CenterCrop(imagesize),
        #     transforms.ToTensor(),
        # ]
        # self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, image_height, image_width)

    def __getitem__(self, idx):
        anomaly, image_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        return {
            "image": image,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths = {}
        path = os.path.join(self.source, self.split.value)
        # maskpath = os.path.join(self.source, "ground_truth") not using ground truth masks
        anomaly_types = os.listdir(path)

        for anomaly in anomaly_types:   # for the case of solar dataset, it will only be anomaly and good types
            anomaly_path = os.path.join(path, anomaly)
            anomaly_files = sorted(os.listdir(anomaly_path))
            imgpaths[anomaly] = [
                os.path.join(anomaly_path, x) for x in anomaly_files
            ]

            if self.train_val_split < 1.0:
                n_images = len(imgpaths[anomaly])
                train_val_split_idx = int(n_images * self.train_val_split)
                if self.split == DatasetSplit.TRAIN:
                    imgpaths[anomaly] = imgpaths[anomaly][:train_val_split_idx]
                elif self.split == DatasetSplit.VAL:
                    imgpaths[anomaly] = imgpaths[anomaly][train_val_split_idx:]

            # comented out because we are not going to use ground truth masks
            # if self.split == DatasetSplit.TEST and anomaly != "good":
            #     anomaly_mask_path = os.path.join(maskpath, anomaly)
            #     anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
            #     maskpaths_per_class[classname][anomaly] = [
            #         os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
            #     ]
            # else:
            #     maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for anomaly in sorted(imgpaths.keys()):
            for i, aux_image_path in enumerate(imgpaths[anomaly]):
                data_tuple = [anomaly, aux_image_path]
                # comented out because we are not going to use ground truth masks
                # if self.split == DatasetSplit.TEST and anomaly != "good":
                #     data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                # else:
                #     data_tuple.append(None)
                data_to_iterate.append(data_tuple)

        return imgpaths, data_to_iterate
