from asyncore import read
import os
import cv2
import numpy as np


def read(path):
    result = []

    if os.stat(path).st_size == 0:
        return result

    with open(db_path, 'r') as f:
        line_content = f.readline()

        for i in range(int(line_content)):
            line_content = f.readline()
            contents = line_content.split(" ")
            name = contents[0]

            row = int(contents[1])
            col = int(contents[2])

            descriptors = []
            for j in range(row):
                values = list(map(np.float32, f.readline().split(" ")))
                descriptors.append(values)

            res.append([name, np.array(descriptors)])

    return result
