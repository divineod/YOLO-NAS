import random
import pickle
import os
import json
import shutil
from sklearn.model_selection import train_test_split
from typing import List
from datasets.innoterra import InnoterraDataset
import torchvision.transforms.functional as TF


def copy_images(source_folder: str, dest_folder: str, file_names: List[str]):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for file_name in file_names:
        shutil.copy(os.path.join(source_folder, file_name), dest_folder)


def save_transformed_images(dataset, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for idx in range(len(dataset)):
        img, _, _ = dataset[idx]
        img = TF.to_pil_image(img)
        img_file_name = dataset.image_file_names[idx]
        img.save(os.path.join(dest_folder, img_file_name))


def split_json(original_json_path: str, image_lists: dict, annotations_folder: str, mode: str ="single"):
    with open(original_json_path, 'r') as json_file:
        data = json.load(json_file)

    annotations_to_remove = []
    for ann in data['annotations']:

        if "bbox" in ann:
            # Calculate area
            _, _, width, height = ann['bbox']
            ann['area'] = width * height
        else:
            annotations_to_remove.append(ann)

        if mode == "single":
            if ann["category_id"] in [3, 4, 5, 6,
                                      7]:  # TODO Divine Comment out after running single defect class experiment
                ann["category_id"] = 3

    categories = []

    if mode == "single":
        categories = [{'id': 3, 'name': 'defect'}]
    elif mode == "multi":
        categories = [{'id': 3, 'name': 'old bruise'}, {'id': 4, 'name': 'old scar'},
                       {'id': 5, 'name': 'new bruise'}, {'id': 6, 'name': 'new scar'},
                       {'id': 7, 'name': 'unclassified bruise or scar'}]

    # Remove the annotations without 'bbox'
    data['annotations'] = [ann for ann in data['annotations'] if ann not in annotations_to_remove]

    # Split the data based on the images in each set
    for set_name, file_names in image_lists.items():
        set_data = {
            'images': [image for image in data['images'] if image['file_name'] in file_names],
            'annotations': [ann for ann in data['annotations'] if
                            ann['image_id'] in [image['id'] for image in data['images'] if
                                                image['file_name'] in file_names]],
            'categories': categories
        }

        with open(os.path.join(annotations_folder, f'{set_name}.json'), 'w') as outfile:
            json.dump(set_data, outfile)

def load_datasets(dataset_folder,
                  images_folder,
                  annotations_folder,
                  dataset: str = "innoterra", seed: int = 42, fixed_splits_id=None,
                  train_defect_ignore_pad_px: int = 0,
                  val_defect_ignore_pad_px: int = 0,
                  **kwargs
                  ):
    """Returns a train and validation dataset."""
    if dataset == 'innoterra':
        all_indexes = list(range(InnoterraDataset(**kwargs).n_samples_total()))

        if fixed_splits_id is None:
            train_indexes, val_indexes = train_test_split(all_indexes, test_size=0.2, random_state=seed)
        else:
            # divide all indexes in 5 equal parts
            assert fixed_splits_id < 5
            random.seed(seed)
            random.shuffle(all_indexes)
            val_indexes = all_indexes[fixed_splits_id::5]
            train_indexes = list(set(all_indexes) - set(val_indexes))

        train_dataset = InnoterraDataset(sample_ids=train_indexes, augment=True,
                                         color_augment=True, defect_ignore_pad_px=train_defect_ignore_pad_px,
                                         resolution=640, **kwargs)
        val_dataset = InnoterraDataset(sample_ids=val_indexes, augment=False,
                                       color_augment=False, defect_ignore_pad_px=val_defect_ignore_pad_px,
                                       resolution=640, **kwargs)
        training_files = train_dataset.image_file_names
        validation_files = val_dataset.image_file_names

        save_transformed_images(train_dataset, os.path.join(dataset_folder, 'train')) # @manuel this splits the image files in the training set to train and val folders
        save_transformed_images(val_dataset, os.path.join(dataset_folder, 'val'))

        # copy_images(images_folder, os.path.join(dataset_folder, 'train'), training_files)
        # copy_images(images_folder, os.path.join(dataset_folder, 'val'), validation_files)

        split_json(os.path.join('/datasets/innoterra_segmentation/', 'annotations.json'), {
            'train': training_files,
            'val': validation_files
        }, annotations_folder, mode="single") # @manuel this one splits the json annotation files into separate folders.

    else:
        raise NotImplementedError(f"Unknown dataset {dataset}")

    return train_dataset, val_dataset


if __name__ == '__main__':

    dataset_folder = '/datasets/innoterra_segmentation/Dataset'  # Base folder path for the dataset
    images_folder = os.path.join('/datasets/innoterra_segmentation/',
                                 'resized_images_1024')  # Folder where the original images are stored
    annotations_folder = '/datasets/innoterra_segmentation/Dataset/annotations'  # Folder where the annotations JSON files will be stored


    for _i in range(5):
        ds_t, ds_v = load_datasets(dataset_folder, images_folder, annotations_folder,"innoterra", 42, fixed_splits_id=_i)
        print(len(ds_t), len(ds_v))

        for t_img_name in ds_t.image_file_names:
            assert t_img_name not in ds_v.image_file_names