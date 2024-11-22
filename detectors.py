import os
from enum import Enum, auto
from multiprocessing import Pool
from typing import List, Dict

import cv2
import numpy as np
from matplotlib import pyplot as plt


# Enum for augmentation types
class AugmentationType(Enum):
    ROTATE_30 = auto()
    ROTATE_70 = auto()
    SCALE_0_5 = auto()
    SCALE_2_0 = auto()
    BLUR = auto()
    NOISE_25 = auto()
    NOISE_50 = auto()


# Enum for detector types
class DetectorType(Enum):
    HARRIS = auto()
    FAST = auto()
    SIFT = auto()
    ORB = auto()


# Function to detect keypoints using the specified type
def detect_keypoints(image: np.ndarray, detector_type: DetectorType) -> List[cv2.KeyPoint]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if detector_type == DetectorType.HARRIS:
        harris = cv2.cornerHarris(gray, blockSize=3, ksize=3, k=0.04)
        harris = cv2.dilate(harris, None)
        keypoints = np.argwhere(harris > 0.01 * harris.max())
        return [cv2.KeyPoint(float(x[1]), float(x[0]), 1) for x in keypoints]

    elif detector_type == DetectorType.FAST:
        fast = cv2.FastFeatureDetector_create()
        fast.setThreshold(20)
        fast.setNonmaxSuppression(True)
        fast.setType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
        return fast.detect(gray, None)

    elif detector_type == DetectorType.ORB:
        orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=4, scoreType=cv2.ORB_HARRIS_SCORE)
        return orb.detect(gray, None)


    elif detector_type == DetectorType.SIFT:
        sift = cv2.SIFT_create()
        return sift.detect(gray, None)

    else:
        raise ValueError("Unsupported detector type.")


# Function to apply augmentations and return the transformation matrix
def apply_augmentations(image: np.ndarray, augmentation: AugmentationType) -> (np.ndarray, np.ndarray):
    def rotate_image(image: np.ndarray, angle: float) -> (np.ndarray, np.ndarray):
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        transformed = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        return transformed, rotation_matrix

    if augmentation == AugmentationType.ROTATE_30:
        return rotate_image(image, 30)

    elif augmentation == AugmentationType.ROTATE_70:
        return rotate_image(image, 70)

    elif augmentation == AugmentationType.SCALE_0_5:
        scale_matrix = np.array([[0.5, 0, 0], [0, 0.5, 0]], dtype=np.float32)
        return (
            cv2.warpAffine(
                image,
                scale_matrix,
                (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)),
            ),
            scale_matrix,
        )

    elif augmentation == AugmentationType.SCALE_2_0:
        scale_matrix = np.array([[2.0, 0, 0], [0, 2.0, 0]], dtype=np.float32)
        return (
            cv2.warpAffine(
                image,
                scale_matrix,
                (int(image.shape[1] * 2.0), int(image.shape[0] * 2.0)),
            ),
            scale_matrix,
        )

    elif augmentation == AugmentationType.BLUR:
        return cv2.GaussianBlur(image, (5, 5), 0), np.eye(3, dtype=np.float32)

    elif augmentation in [AugmentationType.NOISE_25, AugmentationType.NOISE_50]:
        noise_level = 25 if augmentation == AugmentationType.NOISE_25 else 50
        noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
        noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy_image, np.eye(3, dtype=np.float32)

    else:
        return image, np.eye(3, dtype=np.float32)


# Transform keypoints to match the original coordinate system
def transform_keypoints(keypoints: List[cv2.KeyPoint], transformation_matrix: np.ndarray) -> List[cv2.KeyPoint]:
    if not keypoints:
        return []

    points = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
    transformed_points = cv2.transform(points, transformation_matrix)
    return [
        cv2.KeyPoint(x=float(pt[0][0]), y=float(pt[0][1]), size=kp.size)
        for pt, kp in zip(transformed_points, keypoints)
    ]


# Utility function to compute pairwise distances
def compute_pairwise_distances(original: List[cv2.KeyPoint], transformed: List[cv2.KeyPoint]) -> np.ndarray:
    if not original or not transformed:
        return np.array([])

    orig_points = np.array([kp.pt for kp in original])
    trans_points = np.array([kp.pt for kp in transformed])

    # Pairwise distances between original and transformed points
    return np.linalg.norm(orig_points[:, None, :] - trans_points[None, :, :], axis=2)


def plot_results(mean_repeatability: Dict[str, List[float]],
                 mean_localization_error: Dict[str, List[float]],
                 augmentation_labels: List[str]):
    """
    Plots the mean repeatability and localization error as histograms.
    """
    # Extract detector names from the dictionary keys
    detectors = [det.name for det in DetectorType]  # Ensure proper names for detectors

    # Repeatability plot
    plt.figure(figsize=(14, 7))

    # Plot Repeatability
    plt.subplot(1, 2, 1)
    width = 0.2  # Bar width
    x = np.arange(len(augmentation_labels))  # x-axis positions for bars

    for i, detector in enumerate(detectors):
        plt.bar(x + i * width, [mean_repeatability[aug][i] for aug in augmentation_labels], width, label=detector)

    plt.title("Mean Repeatability by Augmentation Type", fontsize=14)
    plt.xlabel("Augmentation Type", fontsize=12)
    plt.ylabel("Repeatability (%)", fontsize=12)
    plt.xticks(x + (len(detectors) - 1) * width / 2, augmentation_labels, rotation=45, ha="right", fontsize=10)
    plt.legend(title="Detectors", fontsize=10)

    # Plot Localization Error
    plt.subplot(1, 2, 2)
    for i, detector in enumerate(detectors):
        plt.bar(x + i * width, [mean_localization_error[aug][i] for aug in augmentation_labels], width, label=detector)

    plt.title("Mean Localization Error by Augmentation Type", fontsize=14)
    plt.xlabel("Augmentation Type", fontsize=12)
    plt.ylabel("Localization Error", fontsize=12)
    plt.xticks(x + (len(detectors) - 1) * width / 2, augmentation_labels, rotation=45, ha="right", fontsize=10)
    plt.legend(title="Detectors", fontsize=10)

    # Finalize and Show
    plt.tight_layout()
    plt.savefig('results_plot.png', dpi=300)  # Save the plot as a high-resolution image
    plt.show()


def calc_metrics(keypoints_original, keypoints_transformed):
    # Evaluate repeatability and localization error
    distances = compute_pairwise_distances(keypoints_original, keypoints_transformed)
    if distances.size == 0:
        repeatability = 0.0
        localization_error = float("inf")
    else:
        # Calculate repeatability
        matches = np.any(distances <= 5.0, axis=1).sum()
        repeatability = round(matches / max(len(keypoints_original), len(keypoints_transformed)), 2)

        # Calculate localization error
        min_distances = np.min(distances, axis=1)
        localization_error = round(float(np.mean(min_distances)), 2)

    return repeatability, localization_error


def load_image(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (100, 100))  # # Resize for optimization
    return image


def process_image(image_path: str, augmentations: list, detectors: list) -> Dict:
    """
    Process a single image with all augmentations and detectors.
    """
    results = {"repeatability": {}, "localization_error": {}}
    image = load_image(image_path)

    for augmentation in augmentations:
        # Apply augmentation
        transformed_image, transformation_matrix = apply_augmentations(image, augmentation)

        for detector in detectors:
            print(image_path, augmentation, detector)

            # Detect keypoints in the original and transformed images
            keypoints_original = detect_keypoints(image, detector)
            keypoints_transformed = detect_keypoints(transformed_image, detector)

            # Transform keypoints back to the original coordinate system
            keypoints_transformed = transform_keypoints(keypoints_transformed, transformation_matrix)

            # Evaluate repeatability and localization error & store results
            repeatability, localization_error = calc_metrics(keypoints_original, keypoints_transformed)
            results["repeatability"][(augmentation.name, detector.name)] = repeatability
            results["localization_error"][(augmentation.name, detector.name)] = localization_error

    return results


def main():
    image_dir = "images"
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.png')]

    augmentations = list(AugmentationType)
    detectors = list(DetectorType)

    all_results = []
    for image_path in image_paths:
        result = process_image(image_path, augmentations, detectors)
        all_results.append(result)

    # Aggregate results
    repeatability = {aug.name: {det.name: [] for det in detectors} for aug in augmentations}
    localization_error = {aug.name: {det.name: [] for det in detectors} for aug in augmentations}

    for result in all_results:
        for key, value in result["repeatability"].items():
            aug, det = key
            repeatability[aug][det].append(value)

        for key, value in result["localization_error"].items():
            aug, det = key
            localization_error[aug][det].append(value)

    # Compute mean values for each augmentation and detector
    mean_repeatability = {
        aug: [np.mean(repeatability[aug][det.name]) for det in detectors]
        for aug in repeatability
    }
    mean_localization_error = {
        aug: [np.mean(localization_error[aug][det.name]) for det in detectors]
        for aug in localization_error
    }

    # Plot results
    plot_results(mean_repeatability, mean_localization_error, [aug.name for aug in augmentations])


if __name__ == "__main__":
    main()
