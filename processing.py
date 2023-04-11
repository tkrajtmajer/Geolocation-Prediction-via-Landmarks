import cv2
import numpy as np


def show(image, title="debug"):
    cv2.imshow(title, image)
    if cv2.waitKey(0):
        return


def get_processed_frame(im):
    # resize and convert to grayscale
    resized_image = cv2.resize(im, (640, 480))
    # gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # return the blurred image
    return cv2.GaussianBlur(resized_image, (5, 5), 0)


def get_frames(path, frequency):
    print("Processing video...")
    frames = []
    video = cv2.VideoCapture(path)
    print(path, frequency)

    # calculate duration of the video
    seconds = round(video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS))
    print('duration:', seconds)

    n = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frequency = int(video.get(cv2.CAP_PROP_FPS) / frequency)

    frames_taken = 0
    count = 0
    success, frame = video.read()

    while success:
        processed_frame = get_processed_frame(frame)
        frames.append(processed_frame)

        frames_taken += 1
        count += frequency
        video.set(cv2.CAP_PROP_POS_FRAMES, count)
        success, frame = video.read()

    # for i in range(0, n, sample_frequency):
    #     # index = sliding_window(video, sample_frequency, i)
    #     video.set(cv2.CAP_PROP_POS_FRAMES, i)
    #     success, frame = video.read()
    #     if success:
    #         count += 1
    #         processed_frame = get_processed_frame(frame)
    #         frames.append(processed_frame)
    #         # show(processed_frame)

    print("Frames extracted from video:", frames_taken)
    return frames
    pass


def sliding_window(video, frequency, fr):
    total = video.get(cv2.CAP_PROP_FRAME_COUNT)
    left = 0
    right = total

    left_fr = fr - (frequency // 3)
    right_fr = fr + (frequency // 3)

    if left_fr > 0:
        left = left_fr

    if right_fr < total - 1:
        right = right_fr

    index = left + 1
    lowest = left
    smallest = None
    video.set(1, left)
    success, curr_frame = video.read()

    while index < right:
        video.set(1, index)
        last_frame = np.copy(curr_frame)
        success, curr_frame = video.read()
        if success:
            diff = last_frame - curr_frame
            if smallest is None or np.sum(diff) < smallest:
                lowest = index
                smallest = np.sum(diff)
        else:
            break

        index += 1

    return lowest
    pass


