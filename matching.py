import cv2
import numpy as np
from SIFT_transform import *
from database_reader import *


def sift_matching(image, database):
    # how many best matches we are gonna take
    k = 10
    # read the database descriptors
    database_descriptors = read(database)
    key_points, descriptors = make_sift(image)

    result = []
    for des in database_descriptors:
        matcher = cv2.BFMatcher(cv2.NORM_L1, True)
        matches = sorted(matcher.match(descriptors, des[1]), key=lambda x: x.distance)
        distance = list(map(lambda x: x.distance, matches[:k]))
        result.append([des[0], np.sum(distance)])

    result.sort(key=lambda x: x[1])
    return result
