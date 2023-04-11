import cv2
import numpy as np


def make_sift(img, number_to_select=None):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    if number_to_select is None:
        return np.array(keypoints), np.array(descriptors)

    return np.array(keypoints)[:number_to_select], np.array(descriptors)[:number_to_select]
