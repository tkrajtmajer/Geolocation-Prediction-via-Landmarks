import cv2
import numpy as np
from processing import *
from matching import *
import evaluate


def get_name(file):
    print(file)
    names = np.array(file.split("_"))
    return names[-1:][0].split(".")[0]


# database path
database_path = "resources/db.txt"

# read gt values
ground_truth = evaluate.read_ground_truth(True)

for video in ground_truth[:, 0]:
    # use this video as a test video
    video_path = 'resources/videos/' + video + '.avi'
    frequency = 1
    frames = get_frames(video_path, frequency)
    score = 0

    for frame in frames:
        cont = frame[1]
        match_list = sift_matching(cont, database_path)

        # get the top prediction and compare with the truth label
        name = ''
        prediction = ''
        # name = get_name(frame[0])
        # prediction = get_name(match_list[0][0])

        if name == prediction:
            score += 1

        print(name, prediction, "accuracy =", score / len(frames))
