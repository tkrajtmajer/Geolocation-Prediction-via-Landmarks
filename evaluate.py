import numpy as np

video_path = 'resources/ground_truth.txt'


def read_ground_truth(train=True):
    gt_values = []

    with open(video_path) as f:
        lines = f.readlines()

    if train:
        for i in range(len(lines) // 2):
            split = lines[i].split()
            gt_values.append([split[0], split[1]])

    if not train:
        for i in range(len(lines) // 2, len(lines)):
            split = lines[i].split()
            gt_values.append([split[0], split[1]])

    return np.array(gt_values)


# print((np.array(read_ground_truth())))
# print(read_ground_truth(False))
