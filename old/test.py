# # import pickle
# #
# # with open('/tmp/pycharm_project_595/best_model_predictions_split_0.pkl', 'rb') as fp:
# #     test = pickle.load(fp)
# #
# # print(test)
#
# import json
# from PIL import Image
# from torchvision.transforms import functional as F
# import matplotlib.pyplot as plt
# import cv2
#
#
# def upscale_image_preserving_aspect_ratio(img_np, base_dim=640, target_dim=1024):
#     """
#     Scale an image from base dimensions to target dimensions, preserving aspect ratio.
#
#     Args:
#     - img_np: The image as a numpy array.
#     - base_dim: The size of the shorter side after padding.
#     - target_dim: The target size for the shorter side.
#
#     Returns:
#     - Scaled image as a numpy array.
#     """
#     # Compute the scale factor for the shorter side
#     scale_factor = target_dim / base_dim
#
#     # Compute new dimensions preserving the aspect ratio
#     new_height = int(img_np.shape[0] * scale_factor)
#     new_width = int(img_np.shape[1] * scale_factor)
#
#     print(f"The new height is {new_height}")
#     print(f"The new width is {new_width}")
#
#     # Resize the image using computed dimensions
#     resized_img = cv2.resize(img_np, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
#
#     return resized_img
#
#
# def upscale_bboxes_preserving_aspect_ratio(bboxes, base_dim=640, target_dim=1024):
#     """
#     Scale bounding box coordinates from original to target dimensions,
#     preserving aspect ratio.
#
#     Args:
#     - bboxes: List of bounding box coordinates [x_min, y_min, x_max, y_max].
#     - base_dim: The size of the shorter side after padding.
#     - target_dim: The target size for the shorter side.
#
#     Returns:
#     - Scaled bounding boxes.
#     """
#     # Compute the scale factor for the shorter side
#     scale_factor = target_dim / base_dim
#
#     scaled_bboxes = []
#     for bbox in bboxes:
#         # Scale each bounding box coordinate with the same scale factor
#         scaled_bbox = [
#             coord * scale_factor for coord in bbox
#         ]
#         scaled_bboxes.append(scaled_bbox)
#
#     return scaled_bboxes
#
#
# def plot_predictions_on_image(img, predictions, class_names):
#     """
#     Plots the model predictions on the image.
#
#     Args:
#     - img: The image as a numpy array.
#     - predictions: The dictionary containing the bounding boxes and class indexes.
#     - class_names: The list of class names.
#     """
#     for bbox, class_idx, confidence in zip(predictions['bboxes'], predictions['class_name_indexes'],
#                                            predictions['confidences']):
#         # Draw the bounding box
#         start_point = (int(bbox[0]), int(bbox[1]))
#         end_point = (int(bbox[2]), int(bbox[3]))
#         color = (255, 0, 0)  # Blue color in BGR
#         thickness = 2
#         img = cv2.rectangle(img, start_point, end_point, color, thickness)
#
#         # Put the class name and confidence
#         label = f"{class_names[class_idx]}: {confidence:.2f}"
#         label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#         top_left = (start_point[0], start_point[1] - label_size[1])
#         bottom_right = (start_point[0] + label_size[0], start_point[1])
#         cv2.rectangle(img, top_left, bottom_right, color, cv2.FILLED)
#         cv2.putText(img, label, (start_point[0], start_point[1] - base_line), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
#                     1)
#
#     # Convert to RGB for plotting
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.imshow(img_rgb)
#     plt.axis('off')  # Hide the axis
#     plt.show()
#
# def load_model_predictions(file_path):
#     with open(file_path, 'r') as f:
#         all_predictions = json.load(f)
#     return all_predictions
#
# file_path = '/tmp/pycharm_project_595/model_predictions/best_model_predictions_split_0.json'  # Replace with your actual file path
# imag_path = "/datasets/innoterra_segmentation/Dataset/val/India_New Bruise_26.jpg"
# all_predictions = load_model_predictions(file_path)
#
# print(all_predictions)


import json
import matplotlib.pyplot as plt
import cv2


def upscale_bboxes_preserving_aspect_ratio(bboxes, base_dim=640, target_dim=1024):
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


def adjust_bboxes_for_upscaled_and_padded_image(bboxes, original_dim=(640, 640), upscaled_padded_dim=(1024, 1024)):
    """
    Adjust bounding boxes for an image that has been upscaled to have its smaller
    side equal to 1024 and then padded to form a 1024x1024 square image.

    Args:
    - bboxes: List of bounding box coordinates [x_min, y_min, x_max, y_max].
    - original_dim: Tuple of the original image dimensions (width, height).
    - upscaled_padded_dim: Tuple of the dimensions after upscaling and padding.

    Returns:
    - Adjusted bounding boxes matching the transformed image dimensions.
    """
    original_width, original_height = original_dim
    upscaled_padded_width, upscaled_padded_height = upscaled_padded_dim

    # Determine scale factor to upscale original image's smaller side to 1024
    scale_factor = min(upscaled_padded_width / original_width, upscaled_padded_height / original_height)

    # Calculate new dimensions of the image after scaling (before padding)
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Determine padding added to make the image a 1024x1024 square
    pad_width = (upscaled_padded_width - new_width) // 2
    pad_height = (upscaled_padded_height - new_height) // 2

    adjusted_bboxes = []
    for bbox in bboxes:
        # Scale bounding boxes
        scaled_x_min = bbox[0] * scale_factor
        scaled_y_min = bbox[1] * scale_factor
        scaled_x_max = bbox[2] * scale_factor
        scaled_y_max = bbox[3] * scale_factor

        # Adjust for padding
        padded_x_min = scaled_x_min + pad_width
        padded_y_min = scaled_y_min + pad_height
        padded_x_max = scaled_x_max + pad_width
        padded_y_max = scaled_y_max + pad_height

        adjusted_bboxes.append([padded_x_min, padded_y_min, padded_x_max, padded_y_max])

    return adjusted_bboxes


def upscale_and_pad_bboxes(bboxes, img_path, base_dim=640, target_dim=1024):
    """
    Scale and pad bounding box coordinates to match an image that has been upscaled
    to 1024 on its shorter side and then padded to form a 1024x1024 square.

    Args:
    - bboxes: List of bounding box coordinates [x_min, y_min, x_max, y_max].
    - img_path: Path to the reference image used to determine scaling and padding.
    - base_dim: The base dimension used for initial upscaling (not used here but kept for consistency).
    - target_dim: The target dimension for the shorter side after upscaling.

    Returns:
    - Adjusted bounding boxes to match the upscaled and padded image dimensions.
    """
    # Load the reference image to determine its original dimensions
    img = cv2.imread(img_path)
    original_height, original_width = img.shape[:2]

    # Determine the scaling factor based on the shorter side
    scale_factor = target_dim / min(original_height, original_width)

    # Calculate new dimensions after scaling
    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)

    # Determine padding needed to make the image square
    if new_width > new_height:
        pad = (new_width - new_height) // 2
        pad_x = 0
        pad_y = pad
    else:
        pad = (new_height - new_width) // 2
        pad_x = pad
        pad_y = 0

    scaled_and_padded_bboxes = []
    for bbox in bboxes:
        # Scale bounding boxes
        scaled_bbox = [coord * scale_factor for coord in bbox]

        # Adjust for padding
        padded_bbox = [
            scaled_bbox[0] + pad_x,
            scaled_bbox[1] + pad_y,
            scaled_bbox[2] + pad_x,
            scaled_bbox[3] + pad_y
        ]
        scaled_and_padded_bboxes.append(padded_bbox)

    return scaled_and_padded_bboxes


def upscale_and_pad_image_one_side(img, target_dim=1024):
    """
    Resize an image to have its longer side match target_dim, then pad it to be square while preserving the aspect ratio,
    with padding applied only to the bottom or right side.

    Args:
    - img: The image as a numpy array.
    - target_dim: The target dimension for the image's longer side.

    Returns:
    - A square image (target_dim x target_dim) with padding applied to one side only.
    """
    original_height, original_width = img.shape[:2]
    # Determine the longer side and calculate scale to make it target_dim
    scale = target_dim / max(original_height, original_width)
    new_height, new_width = int(original_height * scale), int(original_width * scale)

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Calculate padding to make the image square, applying it to one side only
    delta_width = target_dim - new_width
    delta_height = target_dim - new_height

    # Apply padding to the bottom if vertical, to the right if horizontal
    pad_top = 0
    pad_bottom = delta_height
    pad_left = 0
    pad_right = delta_width

    # Apply padding
    square_img = cv2.copyMakeBorder(resized_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                    value=[0, 0, 0])

    return square_img


def plot_predictions_on_image(image_path, predictions, class_names):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting

    # Upscale and pad image
    img = upscale_and_pad_image_one_side(img)

    for bbox, class_idx, confidence in zip(predictions['bboxes'], predictions['class_name_indexes'], predictions['confidences']):
        # Draw the bounding box
        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]), int(bbox[3]))
        color = (255, 0, 0)  # Red color in RGB
        thickness = 2
        img = cv2.rectangle(img, start_point, end_point, color, thickness)

        # Put the class name and confidence
        label = f"{class_names[class_idx]}: {confidence:.2f}"
        cv2.putText(img, label, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    plt.imshow(img)
    plt.axis('off')  # Hide the axis
    plt.show()

def process_predictions(json_file_path, save_path, image_path, annotation_path, target_dim=1024):
    with open(json_file_path, 'r') as file:
        predictions = json.load(file)

    with open(annotation_path, 'r') as file:
        full_annot = json.load(file)

    # Upscale bounding boxes for all predictions
    for pred in predictions:
        # pred["bboxes"] = adjust_bboxes_for_upscaled_and_padded_image(pred["bboxes"])
        pred_id = pred["image_id"]
        img_path = [img["file_name"] for img in full_annot["images"] if img['id'] == pred_id][0]
        img_path = f"/datasets/innoterra_segmentation/resized_images_1024/{img_path}"
        pred["bboxes"] = adjust_bboxes_for_upscaled_and_padded_image(pred["bboxes"])
        # pred["bboxes"] = upscale_and_pad_bboxes(pred["bboxes"], img_path)
        # pred["bboxes"] = upscale_bboxes_preserving_aspect_ratio(pred["bboxes"], target_dim=target_dim)
        plot_predictions_on_image(img_path, pred, pred['class_names'])
    # Save the modified predictions back to the JSON file
    # with open(save_path, 'w') as file:
    #     json.dump(predictions, file, indent=4)

# Path to the JSON file containing the predictions
json_file_path = '/tmp/pycharm_project_595/model_predictions/best_model_predictions_split_0.json'
image_file_path = '/datasets/innoterra_segmentation/resized_images'
annotation_path = '/datasets/innoterra_segmentation/annotations_copy.json'
save_path = '/tmp/pycharm_project_595/model_predictions/upscaled_predictions_split_0.json'

# Process all predictions to upscale bounding boxes
process_predictions(json_file_path, save_path, image_file_path, annotation_path, target_dim=1024)