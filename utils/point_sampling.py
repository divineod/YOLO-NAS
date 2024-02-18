import random
import numpy as np
from scipy.ndimage import distance_transform_edt


def _is_far_enough(new_point, existing_points, min_dist):
    for point in existing_points:
        if np.linalg.norm(np.array(new_point) - np.array(point)) < min_dist:
            return False
    return True


def sample_point_coordinates(mask: np.ndarray, threshold=0.9, min_distance: int = 10, n_points: int = 100,
                             min_point_distance: int = 0,
                             return_distance_mask: bool = False):
    binary_mask = (mask > threshold).astype(np.uint8)
    distance = distance_transform_edt(binary_mask)

    mask_distance_criteria = distance > min_distance

    y, x = np.where(mask_distance_criteria)

    if min_point_distance == 0:
        if len(x) > n_points:  # Ensure there are enough points to sample
            indices = np.random.choice(len(x), n_points, replace=False)
            sampled_points = list(zip(y[indices], x[indices]))
        else:
            print(f"Not enough points meet the criteria to sample {n_points} points.")
            sampled_points = list(zip(y, x))  # Use all available points if less than 100

    else:
        available_points = list(zip(y, x))
        random.shuffle(available_points)
        sampled_points = []
        while len(sampled_points) < n_points and available_points:
            point = available_points.pop()
            if not sampled_points:  # If no points have been added, add the first point directly
                sampled_points.append(point)
            else:
                if _is_far_enough(point, sampled_points, min_point_distance):
                    sampled_points.append(point)
            if len(available_points) == 0:
                print("Ran out of points to check.")
                break

    # switch x and y
    sampled_points = np.array([(point[1], point[0]) for point in sampled_points])

    if return_distance_mask:
        return sampled_points, mask_distance_criteria
    return sampled_points
