import cv2


# Function to compute descriptors and matches
def evaluate_descriptors(image, transformed_image, descriptor_func):
    keypoints_original, descriptors_original = descriptor_func(image)
    keypoints_transformed, descriptors_transformed = descriptor_func(transformed_image)

    # Use BFMatcher for matching
    bf = cv2.BFMatcher()
    matches = bf.match(descriptors_original, descriptors_transformed)

    # Calculate matching accuracy
    accuracy = (
        len(matches) / min(len(descriptors_original), len(descriptors_transformed))
        if min(len(descriptors_original), len(descriptors_transformed)) > 0
        else 0
    )
    return accuracy, len(matches)


def sift_descriptor(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def orb_descriptor(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors
