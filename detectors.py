import os
from enum import Enum, auto
from typing import List, Dict

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


# Enum for augmentation types
class AugmentationType(Enum):
    ROTATE_30 = auto()
    ROTATE_70 = auto()
    SCALE_0_5 = auto()
    SCALE_2_0 = auto()
    BLUR = auto()
    NOISE_5 = auto()
    NOISE_10 = auto()


# Enum for detector types
class DetectorType(Enum):
    HARRIS = auto()
    FAST = auto()
    SIFT = auto()
    ORB = auto()


# Enum for descriptor types
class DescriptorType(Enum):
    SIFT = auto()
    ORB = auto()


# Function to detect keypoints using the specified type
def detect_keypoints(image, detector_type) -> List[cv2.KeyPoint]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if detector_type == DetectorType.HARRIS:
        harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        # harris = cv2.dilate(harris, None)
        keypoints = np.argwhere(harris > 0.01 * harris.max())
        return [cv2.KeyPoint(float(x[1]), float(x[0]), 1) for x in keypoints]

    elif detector_type == DetectorType.FAST:
        fast = cv2.FastFeatureDetector_create()
        return fast.detect(gray, None)

    elif detector_type == DetectorType.ORB:
        orb = cv2.ORB_create()
        return orb.detect(gray, None)

    elif detector_type == DetectorType.SIFT:
        sift = cv2.SIFT_create()
        return sift.detect(gray, None)

    else:
        raise ValueError("Unsupported detector type.")


# Function to apply augmentations and return the transformation matrix
def apply_augmentations(image, augmentation):
    def _rotate_image(angle):
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        return cv2.warpAffine(image, rotation_matrix, image.shape[:2]), rotation_matrix

    def _scale_image(scale_factor):
        scale_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0]], dtype=np.float32)
        new_size = (
            int(image.shape[1] * scale_factor),
            int(image.shape[0] * scale_factor),
        )
        return cv2.warpAffine(image, scale_matrix, new_size), scale_matrix

    def _add_noise(noise_level):
        noise = np.random.normal(0, noise_level**0.5, image.shape).astype(np.int16)
        noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy_image, None

    if augmentation == AugmentationType.ROTATE_30:
        return _rotate_image(30)
    elif augmentation == AugmentationType.ROTATE_70:
        return _rotate_image(70)
    elif augmentation == AugmentationType.SCALE_0_5:
        return _scale_image(0.5)
    elif augmentation == AugmentationType.SCALE_2_0:
        return _scale_image(2.0)
    elif augmentation == AugmentationType.BLUR:
        return cv2.GaussianBlur(image, (5, 5), 0), None
    elif augmentation == AugmentationType.NOISE_5:
        return _add_noise(5)
    elif augmentation == AugmentationType.NOISE_10:
        return _add_noise(10)
    else:
        return image, None


def transform_keypoints(keypoints, transformation_matrix) -> List[cv2.KeyPoint]:
    if not keypoints or transformation_matrix is None:
        return keypoints

    # Transform the points using the forward matrix
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
    transformed_points = cv2.transform(points, transformation_matrix)

    # Create new keypoints with transformed coordinates
    return [cv2.KeyPoint(x=float(pt[0][0]), y=float(pt[0][1]), size=kp.size) for pt, kp in zip(transformed_points, keypoints)]


# Utility function to compute pairwise distances
def compute_pairwise_distances(original, transformed) -> np.ndarray:
    if not original or not transformed:
        return np.array([])

    orig_points = np.array([kp.pt for kp in original])
    trans_points = np.array([kp.pt for kp in transformed])

    # Pairwise distances between original and transformed points
    return np.linalg.norm(orig_points[:, None, :] - trans_points[None, :, :], axis=2)


def plot_results(
    mean_repeatability,
    mean_localization_error,
):
    """
    Plots the mean repeatability and localization error as histograms.
    """
    augmentations = [aug.name for aug in AugmentationType]
    detectors = [det.name for det in DetectorType]

    # Repeatability plot
    plt.figure(figsize=(14, 7))

    # Plot Repeatability
    plt.subplot(1, 2, 1)
    width = 0.2  # Bar width
    x = np.arange(len(augmentations))  # x-axis positions for bars

    for i, detector in enumerate(detectors):
        plt.bar(
            x + i * width,
            [mean_repeatability[aug][i] for aug in augmentations],
            width,
            label=detector,
        )

    plt.title("Mean Repeatability by Augmentation Type", fontsize=14)
    plt.xlabel("Augmentation Type", fontsize=12)
    plt.ylabel("Repeatability (%)", fontsize=12)
    plt.xticks(
        x + (len(detectors) - 1) * width / 2,
        augmentations,
        rotation=45,
        ha="right",
        fontsize=10,
    )
    plt.legend(title="Detectors", fontsize=10)

    # Plot Localization Error
    plt.subplot(1, 2, 2)
    for i, detector in enumerate(detectors):
        plt.bar(
            x + i * width,
            [mean_localization_error[aug][i] for aug in augmentations],
            width,
            label=detector,
        )

    plt.title("Mean Localization Error by Augmentation Type", fontsize=14)
    plt.xlabel("Augmentation Type", fontsize=12)
    plt.ylabel("Localization Error", fontsize=12)
    plt.xticks(
        x + (len(detectors) - 1) * width / 2,
        augmentations,
        rotation=45,
        ha="right",
        fontsize=10,
    )
    plt.legend(title="Detectors", fontsize=10)

    # Finalize and Show
    plt.tight_layout()
    plt.savefig("results_plot1.png", dpi=300)  # Save the plot as a high-resolution image
    # plt.show()


def calc_metrics(keypoints_original, keypoints_transformed):
    # Evaluate repeatability and localization error
    distances = compute_pairwise_distances(keypoints_original, keypoints_transformed)
    if distances.size == 0:
        repeatability = 0.0
        localization_error = float("inf")
    else:
        threshold = 5.0

        # Find matches (points within threshold)
        matches_mask = distances <= threshold
        matches = np.any(matches_mask, axis=1)

        # Calculate repeatability
        repeatability = round(matches.sum() / max(len(keypoints_original), len(keypoints_transformed)), 2) * 100

        # Calculate localization error only for matched points
        matched_distances = distances[matches_mask]  # Get distances only for matched points
        localization_error = round(float(np.mean(matched_distances)), 2) if matched_distances.size > 0 else float("inf")

    return repeatability, localization_error


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return image


def visualize_keypoints(image, keypoints, title):
    """
    Visualizes the provided keypoints on the given image.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    for kp in keypoints:
        x, y = kp.pt
        ax.plot(x, y, "ro", markersize=5)
    ax.set_title(title)
    output_dir = "plots1"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{title}.png"))
    plt.close()


def process_image(image_path, pbar):
    """
    Process a single image with all augmentations and detectors.
    """
    augmentations, detectors = list(AugmentationType), list(DescriptorType)
    results = {"repeatability": {}, "localization_error": {}}
    image = load_image(image_path)

    for augmentation in augmentations:
        # Apply augmentation
        transformed_image, transformation_matrix = apply_augmentations(image, augmentation)

        for detector in detectors:
            # Detect keypoints in the original and transformed images
            keypoints_original = detect_keypoints(image, detector)
            keypoints_transformed = detect_keypoints(transformed_image, detector)

            # Transform original keypoints into the transformed image's coordinate system
            if transformation_matrix is not None:
                keypoints_original_in_transformed_coords = transform_keypoints(keypoints_original, transformation_matrix)
            else:
                keypoints_original_in_transformed_coords = keypoints_original

            # Evaluate repeatability and localization error between transformed keypoints and detected keypoints
            repeatability, localization_error = calc_metrics(keypoints_original_in_transformed_coords, keypoints_transformed)

            # Store results
            results["repeatability"][(augmentation.name, detector.name)] = repeatability
            results["localization_error"][(augmentation.name, detector.name)] = localization_error

            visualize_keypoints(
                image,
                keypoints_original,
                f"{os.path.basename(image_path.split(".")[0])}_{detector.name}",
            )
            visualize_keypoints(
                transformed_image,
                keypoints_transformed,
                f"{os.path.basename(image_path.split(".")[0])}_{detector.name}_{augmentation.name}",
            )

            # Update the progress bar after completing one detector for one augmentation
            pbar.update(1)

    return results


def aggregate_results(results):
    repeatability = {}
    localization_error = {}

    for result in results:
        for aug_det, value in result["repeatability"].items():
            repeatability.setdefault(aug_det, []).append(value)

        for aug_det, value in result["localization_error"].items():
            localization_error.setdefault(aug_det, []).append(value)

    mean_repeatability = {k: np.mean(v) for k, v in repeatability.items()}
    mean_localization_error = {k: np.mean(v) for k, v in localization_error.items()}

    return mean_repeatability, mean_localization_error


def main():
    image_dir = "images1"
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]

    total_tasks = len(image_paths) * len(list(AugmentationType)) * len(list(DetectorType))
    with tqdm(total=total_tasks, desc="Processing All Tasks", unit="task") as pbar:
        all_results = []
        for image_path in image_paths:
            result = process_image(image_path, pbar)
            all_results.append(result)

    mean_repeatability, mean_localization_error = aggregate_results(all_results)
    plot_results(mean_repeatability, mean_localization_error)


if __name__ == "__main__":
    main()
