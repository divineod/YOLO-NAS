import pickle
import json
from PIL import Image
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import cv2
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.datasets.detection_datasets.coco_format_detection import COCOFormatDetectionDataset
from super_gradients.training.transforms.transforms import DetectionMosaic, DetectionRandomAffine, DetectionHSV, \
    DetectionHorizontalFlip, DetectionPaddedRescale, DetectionStandardize, DetectionTargetsFormatTransform, DetectionRescale
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

from super_gradients.training.datasets.datasets_utils import worker_init_reset_seed
from super_gradients.training.utils.collate_fn import CrowdDetectionCollateFN
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training import dataloaders
from super_gradients.training import Trainer
from super_gradients.training import models
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from old.vizualizer import Vizualizer
from datasets import load_datasets
import argparse
import torch
import time
import json
import yaml
import os
import numpy as np
from typing import Tuple

class CustomTransform:
    """This class applies a custom transformation to the image and its annotations."""
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, image, target):
        # Convert numpy array to PIL Image if it's not already
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Resize the image
        image = F.resize(image, self.new_size)

        # Assuming target is a dictionary with 'boxes' key containing the bounding boxes
        if 'boxes' in target:
            # Calculate scale factors
            w_scale = self.new_size[0] / image.width
            h_scale = self.new_size[1] / image.height
            # Scale the bounding box coordinates
            scaled_boxes = []
            for box in target['boxes']:
                x, y, w, h = box
                scaled_boxes.append((x * w_scale, y * h_scale, w * w_scale, h * h_scale))
            target['boxes'] = scaled_boxes

        return image, target


class ResizedCOCOFormatDetectionDataset(COCOFormatDetectionDataset):
    """Child class that resizes images and their annotations to a new size."""
    def __init__(self, root_dir: str, images_subdir: str, annotation_file_path: str, transform=None):
        super().__init__(root_dir, images_subdir, annotation_file_path)
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[any, np.ndarray, np.ndarray]:
        image, annotations, crowd_annotations = super().__getitem__(index)

        # Apply custom transformation to the image and annotations
        if self.transform:
            image, annotations = self.transform(image, annotations)

        # Convert image to tensor
        image = ToTensor()(image)

        return image, annotations, crowd_annotations


def custom_collate_fn(batch):
    # Convert images to tensors if they are NumPy arrays
    images = [torch.from_numpy(item[0]) if isinstance(item[0], np.ndarray) else item[0] for item in batch]

    # Stack the images into a single tensor
    images = torch.stack(images, 0)

    # Targets are a list of tensors of varying sizes, so we leave them as a list
    targets = [item[1] for item in batch]

    return images, targets


def upscale_image(img_np, original_dim=(640, 640), target_dim=(1024, 1024)):
    """
    Scale an image from original to target dimensions.

    Args:
    - img_np: The image as a numpy array.
    - original_dim: Tuple of the original dimensions (width, height).
    - target_dim: Tuple of the target dimensions (width, height).

    Returns:
    - Scaled image as a numpy array.
    """
    return cv2.resize(img_np, target_dim, interpolation=cv2.INTER_LINEAR)


def upscale_bboxes(bboxes, original_dim=(640, 640), target_dim=(1024, 1024)):
    """
    Scale bounding box coordinates from original to target dimensions.

    Args:
    - bboxes: List of bounding box coordinates [x_min, y_min, x_max, y_max].
    - original_dim: Tuple of the original dimensions (width, height).
    - target_dim: Tuple of the target dimensions (width, height).

    Returns:
    - Scaled bounding boxes.
    """
    scale_x = target_dim[0] / original_dim[0]
    scale_y = target_dim[1] / original_dim[1]

    scaled_bboxes = []
    for bbox in bboxes:
        scaled_bbox = [
            bbox[0] * scale_x,
            bbox[1] * scale_y,
            bbox[2] * scale_x,
            bbox[3] * scale_y
        ]
        scaled_bboxes.append(scaled_bbox)
    return scaled_bboxes


def upscale_image_preserving_aspect_ratio(img_np, base_dim=640, target_dim=1024):
    """
    Scale an image from base dimensions to target dimensions, preserving aspect ratio.

    Args:
    - img_np: The image as a numpy array.
    - base_dim: The size of the shorter side after padding.
    - target_dim: The target size for the shorter side.

    Returns:
    - Scaled image as a numpy array.
    """
    # Compute the scale factor for the shorter side
    scale_factor = target_dim / base_dim

    # Compute new dimensions preserving the aspect ratio
    new_height = int(img_np.shape[0] * scale_factor)
    new_width = int(img_np.shape[1] * scale_factor)

    print(f"The new height is {new_height}")
    print(f"The new width is {new_width}")

    # Resize the image using computed dimensions
    resized_img = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_img


def upscale_bboxes_preserving_aspect_ratio(bboxes, base_dim=640, target_dim=1024):
    """
    Scale bounding box coordinates from original to target dimensions,
    preserving aspect ratio.

    Args:
    - bboxes: List of bounding box coordinates [x_min, y_min, x_max, y_max].
    - base_dim: The size of the shorter side after padding.
    - target_dim: The target size for the shorter side.

    Returns:
    - Scaled bounding boxes.
    """
    # Compute the scale factor for the shorter side
    scale_factor = target_dim / base_dim

    scaled_bboxes = []
    for bbox in bboxes:
        # Scale each bounding box coordinate with the same scale factor
        scaled_bbox = [
            coord * scale_factor for coord in bbox
        ]
        scaled_bboxes.append(scaled_bbox)

    return scaled_bboxes


def upscale_and_pad_bboxes(bboxes, original_dim=(640, 640), upscaled_dim=(1024, 1024), final_dim=(1024, 1024)):
    """
    Scale bounding box coordinates to match image upscaling to 1024 on the smaller side,
    followed by padding to reach 1024x1024, preserving the original aspect ratio.

    Args:
    - bboxes: List of bounding box coordinates [x_min, y_min, x_max, y_max].
    - original_dim: Tuple of the original image dimensions (width, height) after initial squaring.
    - upscaled_dim: Tuple of the dimensions after upscaling the smaller side to 1024, before padding.
    - final_dim: Tuple of the final image dimensions (width, height), after padding to 1024x1024.

    Returns:
    - Scaled and padded bounding boxes.
    """
    original_width, original_height = original_dim
    upscaled_width, upscaled_height = upscaled_dim
    final_width, final_height = final_dim

    # Determine the scale factors for upscaling
    scale_x = upscaled_width / original_width
    scale_y = upscaled_height / original_height

    # Calculate padding applied to each side
    pad_x = (final_width - upscaled_width) / 2
    pad_y = (final_height - upscaled_height) / 2

    scaled_and_padded_bboxes = []
    for bbox in bboxes:
        scaled_and_padded_bbox = [
            bbox[0] * scale_x + pad_x,  # x_min scaled and padded
            bbox[1] * scale_y + pad_y,  # y_min scaled and padded
            bbox[2] * scale_x + pad_x,  # x_max scaled and padded
            bbox[3] * scale_y + pad_y   # y_max scaled and padded
        ]
        scaled_and_padded_bboxes.append(scaled_and_padded_bbox)

    return scaled_and_padded_bboxes


def plot_predictions_on_image(img, predictions, class_names):
    """
    Plots the model predictions on the image.

    Args:
    - img: The image as a numpy array.
    - predictions: The dictionary containing the bounding boxes and class indexes.
    - class_names: The list of class names.
    """
    for bbox, class_idx, confidence in zip(predictions['bboxes'], predictions['class_name_indexes'],
                                           predictions['confidences']):
        # Draw the bounding box
        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]), int(bbox[3]))
        color = (255, 0, 0)  # Blue color in BGR
        thickness = 2
        img = cv2.rectangle(img, start_point, end_point, color, thickness)

        # Put the class name and confidence
        label = f"{class_names[class_idx]}: {confidence:.2f}"
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top_left = (start_point[0], start_point[1] - label_size[1])
        bottom_right = (start_point[0] + label_size[0], start_point[1])
        cv2.rectangle(img, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(img, label, (start_point[0], start_point[1] - base_line), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                    1)

    # Convert to RGB for plotting
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide the axis
    plt.show()


def save_model_predictions(model, dataloader, file_path):
    model.eval()  # Ensure model is in evaluation mode
    all_predictions = []

    for i, (imgs, _, _) in enumerate(dataloader):
        batch_sample_ids = range(i * dataloader.batch_size, (i + 1) * dataloader.batch_size)
        batch_coco_ids = [dataloader.dataset.sample_id_to_coco_id[sample_id] for sample_id in batch_sample_ids]

        # Process each image in the batch
        for img_index, img in enumerate(imgs):
            img_np = img.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format and CPU numpy array
            img_np = (img_np * 255).astype(np.uint8)  # Convert to uint8
            img_np_upscaled = upscale_image_preserving_aspect_ratio(img_np)  # Upscale the image
            img_np_upscaled_bgr = cv2.cvtColor(img_np_upscaled, cv2.COLOR_RGB2BGR)
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB format
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            model_predictions = model.predict(img_rgb)

            bboxes = model_predictions.prediction.bboxes_xyxy.tolist()  # Original bounding boxes
            scaled_bboxes = upscale_bboxes_preserving_aspect_ratio(bboxes)  # Scale the bounding boxes

            class_names = model_predictions.class_names
            class_name_indexes = model_predictions.prediction.labels.tolist()
            confidences = model_predictions.prediction.confidence.tolist()

            single_image_prediction = {
                "image_id": batch_coco_ids[img_index],
                "bboxes": scaled_bboxes,
                "class_names": class_names,
                "class_name_indexes": class_name_indexes,
                "confidences": confidences
            }

            # Plot the predictions on the image
            plot_predictions_on_image(img_np_upscaled_bgr, single_image_prediction, class_names)

            all_predictions.append(single_image_prediction)

    # Save the aggregated predictions to a file
    with open(file_path, 'w') as f:
        json.dump(all_predictions, f)

    print(f"Predictions saved to {file_path}")


def find_last_modified_subfolder(run_dir):
    # Find all directories within the run_dir
    sub_dirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]

    if not sub_dirs:
        raise Exception(f"No directories found in {run_dir}")

    # Sort directories by last modification time, newest first
    last_modified_dir = sorted(sub_dirs, key=lambda x: os.path.getmtime(os.path.join(run_dir, x)), reverse=True)[0]

    # Construct the path to ckpt_best.pth in the most recently modified directory
    checkpoint_dir = os.path.join(run_dir, last_modified_dir)
    checkpoint_path = os.path.join(checkpoint_dir, 'ckpt_best.pth')

    return checkpoint_path


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--data", type=str, required=True,
                    help="path to data.yaml")
    ap.add_argument("-n", "--name", type=str,
                    help="Checkpoint dir name")
    ap.add_argument("-b", "--batch", type=int, default=6,
                    help="Training batch size")
    ap.add_argument("-e", "--epoch", type=int, default=100,
                    help="Training number of epochs")
    ap.add_argument("-j", "--worker", type=int, default=2,
                    help="Training number of workers")
    ap.add_argument("-m", "--model", type=str, default='yolo_nas_s',
                    choices=['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'],
                    help="Model type (eg: yolo_nas_s)")
    ap.add_argument("-w", "--weight", type=str, default='coco',
                    help="path to pre-trained model weight")
    ap.add_argument("-r", "--resume", action='store_true',
                    help="to resume model training")
    ap.add_argument("-s", "--size", type=int, default=640,
                    help="input image size")
    ap.add_argument("--gpus", action='store_true',
                    help="Run on all gpus")
    ap.add_argument("--cpu", action='store_true',
                    help="Run on CPU")

    # train_params
    ap.add_argument("--warmup_mode", type=str, default='linear_epoch_step',
                    help="Warmup Mode")
    ap.add_argument("--warmup_initial_lr", type=float, default=1e-6,
                    help="Warmup Initial LR")
    ap.add_argument("--lr_warmup_epochs", type=int, default=3,
                    help="LR Warmup Epochs")
    ap.add_argument("--initial_lr", type=float, default=5e-4,
                    help="Inital LR")
    ap.add_argument("--lr_mode", type=str, default='cosine',
                    help="LR Mode")
    ap.add_argument("--cosine_final_lr_ratio", type=float, default=0.1,
                    help="Cosine Final LR Ratio")
    ap.add_argument("--optimizer", type=str, default='AdamW',
                    help="Optimizer")
    ap.add_argument("--weight_decay", type=float, default=0.0001,
                    help="Weight Decay")
    args = vars(ap.parse_args())

    # Start Time
    s_time = time.time()

    if args['name'] is None:
        name = 'train'
    else:
        name = args['name']

    if args['resume']:
        name = os.path.split(args['weight'])[0].split('/')[-1]
    else:
        n = 0
        while True:
            if not os.path.exists(os.path.join('runs', f'{name}{n}')):
                name = f'{name}{n}'
                os.makedirs(os.path.join('runs', name))
                break
            else:
                n += 1
    print(f"[INFO] Checkpoints saved in \033[1m{os.path.join('runs', name)}\033[0m")

    # Training on GPU or CPU
    if args['cpu']:
        print('[INFO] Training on \033[1mCPU\033[0m')
        trainer = Trainer(experiment_name=name, ckpt_root_dir='runs', device='cpu')
    elif args['gpus']:
        print(f'[INFO] Training on GPU: \033[1m{torch.cuda.get_device_name()}\033[0m')
        trainer = Trainer(experiment_name=name, ckpt_root_dir='runs', multi_gpu=args['gpus'])
    else:
        print(f'[INFO] Training on GPU: \033[1m{torch.cuda.get_device_name()}\033[0m')
        trainer = Trainer(experiment_name=name, ckpt_root_dir='runs')

    dataset_folder = '/datasets/innoterra_segmentation/Dataset'  # Base folder path for the dataset
    images_folder = os.path.join('/datasets/innoterra_segmentation/',
                                 'resized_images_1024')  # Folder where the original images are stored
    annotations_folder = '/datasets/innoterra_segmentation/Dataset/annotations'  # Folder where the annotations JSON files will be stored

    for _i in range(5):
        print(f"Running evaluation on valset {_i}")

        train_dataset, val_dataset = load_datasets(
            dataset_folder, images_folder, annotations_folder,
            annotation_type="bboxes",
            separate_background_banana=False,
            separate_defect_types=False,
            seed=42,  # shuffles filename list before splitting
            fixed_splits_id=_i  # if not none, ID of 5-fold split (can be 0,1,2,3,4)
        )

        # Load Path Params
        yaml_params = yaml.safe_load(open(args['data'], 'r'))
        with open(os.path.join(yaml_params['Dir'], yaml_params['labels']['train'])) as f:
            no_class = len(json.load(f)['categories'])
            f.close()
        print(f"\033[1m[INFO] Number of Classes: {no_class}\033[0m")

        #  Code block to store the file_names after each training iteration
        train_filenames = train_dataset.image_file_names
        val_filenames = val_dataset.image_file_names
        train_base_filename = 'train_filenames'
        val_base_filename = 'val_filenames'
        extension = '.json'
        index = 0
        base_dir = '/tmp/pycharm_project_595/train_val_filenames/'
        train_filename = f'{base_dir}{train_base_filename}{extension}'
        val_filename = f'{base_dir}{val_base_filename}{extension}'

        # Check if the file exists
        while os.path.exists(train_filename):
            # Update the filename to include the index if the base filename exists
            train_filename = f'{base_dir}{train_base_filename}_{index}{extension}'
            index += 1

        index = 0
        while os.path.exists(val_filename):
            # Update the filename to include the index if the base filename exists
            val_filename = f'{base_dir}{val_base_filename}_{index}{extension}'
            index += 1

        # Save the file with the new filename
        with open(train_filename, 'w') as fp:
            json.dump(train_filenames, fp)

        print(f'Training set filenames saved as: {train_filename}')

        # Save the file with the new filename
        with open(val_filename, 'w') as fp:
            json.dump(val_filenames, fp)

        print(f'Validation set filenames saved as: {val_filename}')

        # Retrain Dataset
        # trainset = COCOFormatDetectionDataset(data_dir=yaml_params['Dir'],
        #                                       images_dir=yaml_params['images']['train'],
        #                                       json_annotation_file=yaml_params['labels']['train'],
        #                                       input_dim=(args['size'], args['size']),
        #                                       ignore_empty_annotations=False)

        trainset = COCOFormatDetectionDataset(data_dir=yaml_params['Dir'],
                                             images_dir=yaml_params['images']['train'],
                                             json_annotation_file=yaml_params['labels']['train'],
                                             input_dim=(args['size'], args['size']),
                                             ignore_empty_annotations=False,
                                               transforms=[
                                                   DetectionMosaic(prob=1., input_dim=(args['size'], args['size'])),
                                                   DetectionRandomAffine(degrees=0., scales=(0.5, 1.5), shear=0.,
                                                                         target_size=(args['size'], args['size']),
                                                                         filter_box_candidates=False, border_value=128),
                                                   DetectionHSV(prob=1., hgain=5, vgain=30, sgain=30),
                                                   DetectionHorizontalFlip(prob=0.5),
                                                   DetectionPaddedRescale(input_dim=(args['size'], args['size']),
                                                                          max_targets=300),
                                                   DetectionStandardize(max_value=255),
                                                   DetectionTargetsFormatTransform(max_targets=300, input_dim=(
                                                   args['size'], args['size']),
                                                                                   output_format="LABEL_CXCYWH")
                                               ]
                                               )

        train_loader = dataloaders.get(dataset=trainset, dataloader_params={
            "shuffle": True,
            "batch_size": args['batch'],
            "drop_last": False,
            "pin_memory": True,
            "collate_fn": CrowdDetectionCollateFN(),
            "worker_init_fn": worker_init_reset_seed,
            "min_samples": 512
        })

        # Valid Data
        valset = COCOFormatDetectionDataset(data_dir=yaml_params['Dir'],
                                            images_dir=yaml_params['images']['val'],
                                            json_annotation_file=yaml_params['labels']['val'],
                                            input_dim=(args['size'], args['size']),
                                            ignore_empty_annotations=False,
                                            transforms=[
                                                DetectionPaddedRescale(input_dim=(args['size'], args['size']),
                                                                       max_targets=300),
                                                DetectionStandardize(max_value=255),
                                                DetectionTargetsFormatTransform(max_targets=300,
                                                                                input_dim=(args['size'], args['size']),
                                                                                output_format="LABEL_CXCYWH")
                                            ]
                                            )


        valid_loader = dataloaders.get(dataset=valset, dataloader_params={
            "shuffle": False,
            "batch_size": int(args['batch'] * 2),
            "num_workers": args['worker'],
            "drop_last": False,
            "pin_memory": True,
            "collate_fn": CrowdDetectionCollateFN(),
            "worker_init_fn": worker_init_reset_seed
        })

        val_loader = DataLoader(valset, batch_size=int(args['batch'] * 2), shuffle=False, collate_fn=custom_collate_fn)

        # Test Data
        if 'test' in (yaml_params['images'].keys() or yaml_params['labels'].keys()):
            testset = COCOFormatDetectionDataset(data_dir=yaml_params['Dir'],
                                                 images_dir=yaml_params['images']['test'],
                                                 json_annotation_file=yaml_params['labels']['test'],
                                                 input_dim=(args['size'], args['size']),
                                                 ignore_empty_annotations=False,
                                                 transforms=[
                                                     DetectionPaddedRescale(input_dim=(args['size'], args['size']),
                                                                            max_targets=300),
                                                     DetectionStandardize(max_value=255),
                                                     DetectionTargetsFormatTransform(max_targets=300,
                                                                                     input_dim=(args['size'], args['size']),
                                                                                     output_format="LABEL_CXCYWH")
                                                 ])
            test_loader = dataloaders.get(dataset=testset, dataloader_params={
                "shuffle": False,
                "batch_size": int(args['batch'] * 2),
                "num_workers": args['worker'],
                "drop_last": False,
                "pin_memory": True,
                "collate_fn": CrowdDetectionCollateFN(),
                "worker_init_fn": worker_init_reset_seed
            })

        # To Resume Training
        if args['resume']:
            model = models.get(
                args['model'],
                num_classes=no_class,
                checkpoint_path=args["weight"]
            )
        else:
            model = models.get(
                args['model'],
                num_classes=no_class,
                pretrained_weights=args["weight"]
            )

        train_params = {
            'silent_mode': False,
            "average_best_models": True,
            "warmup_mode": args['warmup_mode'],
            "warmup_initial_lr": args['warmup_initial_lr'],
            "lr_warmup_epochs": args['lr_warmup_epochs'],
            "initial_lr": args['initial_lr'],
            "lr_mode": args['lr_mode'],
            "cosine_final_lr_ratio": args['cosine_final_lr_ratio'],
            "optimizer": args['optimizer'],
            "optimizer_params": {"weight_decay": args['weight_decay']},
            "zero_weight_decay_on_bias_and_bn": True,
            "ema": True,
            "ema_params": {"decay": 0.9, "decay_type": "threshold"},
            "max_epochs": args['epoch'],
            "mixed_precision": True,
            "loss": PPYoloELoss(
                use_static_assigner=False,
                num_classes=no_class,
                reg_max=16
            ),
            "valid_metrics_list": [
                DetectionMetrics_050(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=no_class,
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(
                        score_threshold=0.01,
                        nms_top_k=1000,
                        max_predictions=300,
                        nms_threshold=0.7
                    )
                )
            ],
            "metric_to_watch": 'mAP@0.50'
        }

        # to Resume Training
        if args['resume']:
            train_params['resume'] = True

        # Print Training Params
        print('[INFO] Training Params:\n', train_params)

        # Model Training...
        trainer.train(
            model=model,
            training_params=train_params,
            train_loader=train_loader,
            valid_loader=valid_loader
        )
        # visualizer = Vizualizer()
        # plots = visualizer.visualize_predictions(model=model, dataloader=val_loader)

        # Load best model
        base_dir = '/tmp/pycharm_project_595/runs'

        # Construct the path to the specific run directory (e.g., train72)
        run_dir = os.path.join(base_dir, name)
        checkpoint_path = find_last_modified_subfolder(run_dir)

        # Now, you have the correct checkpoint_path that includes the arbitrary folder name
        best_model = models.get(args['model'],
                                num_classes=no_class,
                                checkpoint_path=checkpoint_path)

        # best_model = models.get(args['model'],
        #                         num_classes=no_class,
        #                         checkpoint_path=os.path.join('/tmp/pycharm_project_595/runs', name, 'ckpt_best.pth'))

        save_model_predictions(best_model, valid_loader, f'model_predictions/best_model_predictions_split_{_i}.json')

        # Evaluating on Val Dataset
        eval_model = trainer.test(model=best_model,
                                  test_loader=valid_loader,
                                  test_metrics_list=DetectionMetrics_050(score_thres=0.1,
                                                                         top_k_predictions=300,
                                                                         num_cls=no_class,
                                                                         normalize_targets=True,
                                                                         post_prediction_callback=PPYoloEPostPredictionCallback(
                                                                             score_threshold=0.01,
                                                                             nms_top_k=1000,
                                                                             max_predictions=300,
                                                                             nms_threshold=0.7)
                                                                         ))
        print('\033[1m [INFO] Validating Model:\033[0m')
        for i in eval_model:
            print(f"{i}: {float(eval_model[i])}")

        ###### Commented out because no test sets yet
        # # Evaluating on Test Dataset
        # if 'test' in (yaml_params['images'].keys() or yaml_params['labels'].keys()):
        #     test_result = trainer.test(model=best_model,
        #                                test_loader=test_loader,
        #                                test_metrics_list=DetectionMetrics_050(score_thres=0.1,
        #                                                                       top_k_predictions=300,
        #                                                                       num_cls=no_class,
        #                                                                       normalize_targets=True,
        #                                                                       post_prediction_callback=PPYoloEPostPredictionCallback(
        #                                                                           score_threshold=0.01,
        #                                                                           nms_top_k=1000,
        #                                                                           max_predictions=300,
        #                                                                           nms_threshold=0.7)
        #                                                                       ))
        #     print('\033[1m [INFO] Test Results:\033[0m')
        #     for i in test_result:
        #         print(f"{i}: {float(test_result[i])}")
        # print(f'[INFO] Training Completed in \033[1m{(time.time() - s_time) / 3600} Hours\033[0m')
