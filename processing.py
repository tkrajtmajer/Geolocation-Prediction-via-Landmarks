import cv2
import numpy as np


def show(image, title="debug"):
    cv2.imshow(title, image)
    if cv2.waitKey(0):
        return


def get_processed_frame(im):
    # resize
    new_size = (int(im.shape[1] / 4), int(im.shape[0] / 4))
    resized_image = cv2.resize(im, new_size)
    # convert to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # return the blurred image
    return cv2.GaussianBlur(gray, (5, 5), 0)


def get_frames(path, frequency):
    frames = []
    video = cv2.VideoCapture(path)
    print(path, frequency)

    n = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frequency = int(video.get(cv2.CAP_PROP_FPS) / frequency)

    for i in range(0, n, sample_frequency):
        is_successful, frame = video.read()
        if is_successful:
            processed_frame = get_processed_frame(frame)
            frames.append(processed_frame)
            show(processed_frame)

    return frames

