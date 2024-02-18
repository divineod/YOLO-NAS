import os

import torchvision.ops
import yaml
import json
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T

import torchvision.transforms.v2 as T2
from torchvision import tv_tensors

from scipy.ndimage import binary_dilation
from pycocotools import mask as mask_utils

from utils.visualizer import SegmentationMapVisualizer
from utils.crop_augmentation import get_padding_mask

print(torch.cuda.is_available())
config = yaml.safe_load(open("config.yaml"))


class CustomPadToSquare:
    def __init__(self, target_size, fill=0, padding_mode="constant"):
        self.target_size = target_size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        # Calculate padding
        width, height = F.get_image_size(img)
        max_side = max(width, height)
        pad_width = (self.target_size - width) // 2 if width < max_side else 0
        pad_height = (self.target_size - height) // 2 if height < max_side else 0

        # Apply padding
        padding = (pad_width, pad_height, pad_width, pad_height)
        return F.pad(img, padding, self.fill, self.padding_mode)


class InnoterraDataset(Dataset):
    """
    Image (semantic) segmentation dataset.
    """

    root_dir = config["datasets"]["innoterra"]["root_dir"]
    img_dir = os.path.join(root_dir, "resized_images_1024")
    annotations_path = os.path.join(root_dir, "annotations_copy.json")

    defect_indexes = [3, 4, 5, 6, 7]
    ignore_index = 255

    def __init__(self, sample_ids=None, augment=True, color_augment=True, resolution=1024,
                 defect_ignore_pad_px=0, annotation_type="masks",
                 separate_background_banana=True,
                 separate_defect_types=False):

        if annotation_type == "panoptic" and defect_ignore_pad_px > 0:
            raise NotImplementedError("Panoptic segmentation does not support defect_ignore_pad_px")

        self.augment = augment
        self.resolution = resolution
        self.image_file_names = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        self.defect_ignore_pad_px = defect_ignore_pad_px

        self.separate_background_banana = separate_background_banana
        self.separate_defect_types = separate_defect_types

        assert annotation_type in ["masks", "bboxes", "masks+bboxes", "panoptic"], \
            "annotation_type must be 'masks' or 'bboxes' or 'masks+bboxes', or 'panoptic'"
        self.annotation_type = annotation_type
        self.annotations = json.load(open(self.annotations_path))
        self.filenames_to_ids = {i['file_name']: i['id'] for i in self.annotations['images']}
        self.ids_to_shapes = {i['id']: (i['height'], i['width']) for i in self.annotations['images']}

        if sample_ids is not None:  # used to sample train_test_split
            self.image_file_names = [self.image_file_names[i] for i in sample_ids]

        if augment:
            self.transform = T2.Compose([
                T2.ToImage(),
                T2.ToDtype(torch.float32, scale=True),
                T2.RandomCrop((1024, 1024), pad_if_needed=True),
                T2.Resize(self.resolution, antialias=True),
                T2.RandomHorizontalFlip(0.5)
            ])
        else:
            self.transform = T2.Compose([
                T2.ToImage(),
                T2.ToDtype(torch.float32, scale=True),
                T2.CenterCrop((1024, 1024)),
                T2.Resize(self.resolution, antialias=True),
            ])

        if color_augment:
            CAUG_FACTOR = 0.5
            self.color_transform = T.Compose([
                T.ColorJitter(brightness=0.2 * CAUG_FACTOR,
                              contrast=0.2 * CAUG_FACTOR,
                              saturation=0.2 * CAUG_FACTOR,
                              hue=0.1 * CAUG_FACTOR),
                T.RandomAdjustSharpness(2 * CAUG_FACTOR),
                # T.RandomEqualize(),
                # T.RandomAutocontrast()
            ])
        else:
            self.color_transform = None

        class_names = ["background"]
        if self.separate_background_banana:
            class_names.append("background_banana")
        class_names.append("foreground_banana")
        if self.separate_defect_types:
            class_names.extend(["old bruise", "old scar", "new bruise", "new scar"])
        else:
            class_names.append("defect")
        self.class_names = class_names

    @property
    def banana_ids(self):
        return {1, 2} if self.separate_background_banana else {1}

    @property
    def defect_ids(self):
        if self.separate_defect_types and self.separate_background_banana:
            return {3, 4, 5, 6}
        elif self.separate_defect_types and not self.separate_background_banana:
            return {2, 3, 4, 5}
        elif not self.separate_defect_types and self.separate_background_banana:
            return {3}
        else:
            return {2}

    @property
    def class_dict(self):
        return {i: self.class_names[i] for i in range(len(self.class_names))}

    def _merge_classes(self, x: torch.Tensor) -> torch.Tensor:

        if not self.separate_defect_types:
            x[(x >= 3) & (x < 255)] = 3

        if not self.separate_background_banana:
            # Merging background banana and foreground banana into single class
            x[x == 2] = 1
            x[(x > 1) & (x < 255)] -= 1  # Move the defects class to a new label

        return x

    @classmethod
    def n_samples_total(cls):
        return len([f for f in os.listdir(cls.img_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img_file_name = self.image_file_names[idx]
        image_path = os.path.join(self.img_dir, img_file_name)
        image = Image.open(image_path).convert("RGB")
        segmask, bboxes, instance_mask = None, None, None

        if self.color_transform is not None:
            image = self.color_transform(image)

        if self.annotation_type == "panoptic":
            mask, id_map = self.load_panoptic_mask(img_file_name)
            instance_mask = tv_tensors.Mask(mask)

        if "masks" in self.annotation_type:
            segmask = self._load_segmentation_mask(img_file_name)
        else:
            segmask = None

        if "bboxes" in self.annotation_type:
            bboxes, class_ids = self._load_bboxes(img_file_name, (image.size[1], image.size[0]))
        else:
            bboxes, class_ids = None, None

        image, segmask, bboxes, instance_mask = self.transform(image, segmask, bboxes, instance_mask)

        if self.annotation_type == "masks":
            return image, segmask
        elif self.annotation_type == "bboxes":
            return image, bboxes, class_ids
        elif self.annotation_type == "masks+bboxes":
            return image, segmask, bboxes, class_ids
        elif self.annotation_type == "panoptic":
            padding_mask = get_padding_mask(image)
            return image, padding_mask, instance_mask, id_map

    def _get_segmentation_mask_from_rle(self, image_id: int) -> torch.Tensor:

        out_shape = self.ids_to_shapes[image_id]
        mask = np.zeros(out_shape, dtype=np.uint8)
        relevant_annotations = [a for a in self.annotations['annotations'] if a['image_id'] == image_id]
        for class_id in range(1, 8):
            for ann in relevant_annotations:
                if ann['category_id'] == class_id:
                    rle = ann['segmentation']
                    binary_mask = mask_utils.decode(rle)
                    mask[binary_mask == 1] = class_id
        return mask

    def _get_panoptic_mask_from_rle(self, image_id: int) -> Tuple[torch.Tensor, dict]:
        out_shape = self.ids_to_shapes[image_id]
        mask = np.zeros(out_shape, dtype=np.int64)
        relevant_annotations = [a for a in self.annotations['annotations'] if a['image_id'] == image_id]
        instance_to_class_map = {0: 0}
        for ann in relevant_annotations:
            instance_id = ann['id'] + 1
            class_id = ann['category_id']
            rle = ann['segmentation']
            binary_mask = mask_utils.decode(rle)
            mask[binary_mask == 1] = instance_id
            instance_to_class_map[instance_id] = class_id
        return mask, instance_to_class_map

    def load_panoptic_mask(self, image_filename: str) -> Tuple[torch.Tensor, dict]:
        image_id = self.filenames_to_ids[image_filename]
        mask, id_map = self._get_panoptic_mask_from_rle(image_id)

        # merge classes in dict
        for k, v in id_map.items():
            if not self.separate_background_banana and v == 2:
                id_map[k] = 1
            if not self.separate_defect_types and v > 2 and v != 255:
                id_map[k] = 3 if self.separate_background_banana else 2

        return mask, id_map

    def _load_segmentation_mask(self, image_filename: str) -> torch.Tensor:

        image_id = self.filenames_to_ids[image_filename]
        mask = self._get_segmentation_mask_from_rle(image_id)

        if self.defect_ignore_pad_px > 0:
            defect_binary = np.isin(mask, self.defect_indexes)
            dilated_defect_binary = binary_dilation(defect_binary, iterations=self.defect_ignore_pad_px)
            defect_edges = dilated_defect_binary & ~defect_binary
            mask[defect_edges] = self.ignore_index

        mask = self._merge_classes(mask)

        mask = tv_tensors.Mask(mask)

        return mask

    def _load_bboxes(self, image_filename: str, shape) -> Tuple[torch.Tensor, torch.Tensor]:
        image_id = self.filenames_to_ids[image_filename]
        relevant_annotations = [a for a in self.annotations['annotations'] if a['image_id'] == image_id and "bbox" in a]

        try:
            bboxes = torch.stack([torch.tensor(a['bbox']) for a in relevant_annotations])
            bboxes = torchvision.ops.box_convert(bboxes, 'xywh', 'xyxy')  # convert from xywh to xyxy
            bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=shape)

            class_ids = torch.tensor([a['category_id'] for a in relevant_annotations])

            class_ids = self._merge_classes(class_ids)
        except Exception as e:
            bboxes, class_ids = None, None

        return bboxes, class_ids


if __name__ == '__main__':
    # TEST segmentation
    if False:
        ds = InnoterraDataset(augment=True, defect_ignore_pad_px=5, annotation_type="masks")

        vis = SegmentationMapVisualizer()
        img, mask = ds[0]
        plt.imshow(img.permute(1, 2, 0))
        plt.show()
        mask_vis = vis(mask)
        plt.imshow(mask_vis.permute(1, 2, 0))
        plt.show()

    # TEST objdet
    if False:
        from torchvision.utils import draw_bounding_boxes
        from torchvision.transforms.v2 import functional as F

        ds = InnoterraDataset(annotation_type="bboxes", augment=False)
        img, bboxes, labels = ds[10]
        print(bboxes, labels)

        viz = draw_bounding_boxes((img * 255.).to(torch.uint8), boxes=bboxes)
        F.to_pil_image(viz).show()

    # TEST both
    if False:
        ds = InnoterraDataset(annotation_type="masks+bboxes", augment=True)
        img, mask, bboxes, labels = ds[10]
        print(bboxes, labels)

        vis = SegmentationMapVisualizer()
        mask_vis = vis(mask)
        plt.imshow(mask_vis.permute(1, 2, 0))
        plt.show()

        from torchvision.utils import draw_bounding_boxes
        from torchvision.transforms.v2 import functional as F

        viz = draw_bounding_boxes((img * 255.).to(torch.uint8), boxes=bboxes)
        F.to_pil_image(viz).show()

    # TEST panoptic
    if True:
        ds = InnoterraDataset(annotation_type="panoptic", augment=True)
        img, _, mask, id_map = ds[-1]
        print(id_map)

        vis = SegmentationMapVisualizer(pallette="random")
        mask_vis = vis(mask)
        print(np.unique(mask))
        plt.imshow(mask_vis.permute(1, 2, 0))
        plt.show()

        print(id_map)
