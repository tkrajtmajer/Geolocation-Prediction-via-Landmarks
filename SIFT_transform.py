import cv2


def make_sift(img, number_to_select=None):
    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(img, None)

    if number_to_select is None:
        return keypoints, descriptors

    return keypoints[:number_to_select], descriptors[:number_to_select]
