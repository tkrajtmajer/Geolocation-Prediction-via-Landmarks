import cv2
import numpy as np
from processing import *
from matching import *
import evaluate
import geolocation
import SIFT_transform
import query

folder_path = 'resources/videos/'
db_path = 'resources/db/'
db_name = 'database'

print('START')

q = query.Query()

video_name = 'VIDEO0191.avi'
video_path = 'resources/videos/' + video_name

frames = get_frames(video_path, 10)
score = 0
acc = 0

for frame in frames:
    # img_resize = cv2.resize(frame, (640, 480))
    keypoints, descriptors = SIFT_transform.make_sift(frame)

    chans = cv2.split(frame)
    color_hist = np.zeros((256, len(chans)))
    for i in range(len(chans)):
        color_hist[:, i] = np.histogram(chans[i], bins=np.arange(256 + 1))[0] / float(
            (chans[i].shape[0] * chans[i].shape[1]))


    sift_winners, sift_distances, hist_winners, hist_distances = q.find(descriptors, color_hist)
    print(sift_winners, sift_distances, hist_winners, hist_distances)


"""
for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    print(file_path)
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("Error opening video file")

    while cap.isOpened():
        # Read a frame from the video file
        ret, frame = cap.read()

        # Check if a frame was successfully read
        if not ret:
            break

        # Process the frame
        img_resize = cv2.resize(frame, (640, 480))
        keypoints, descriptors = SIFT_transform.make_sift(img_resize)
        chans = cv2.split(img_resize)
        color_hist = np.zeros((256, len(chans)))
        for i in range(len(chans)):
            color_hist[:, i] = np.histogram(chans[i], bins=np.arange(256 + 1))[0] / float(
                (chans[i].shape[0] * chans[i].shape[1]))

        sift_winners, sift_distances, hist_winners, hist_distances = q.find(descriptors, color_hist)
        print(sift_winners, sift_distances, hist_winners, hist_distances)

        # Display the frame (optional)
        cv2.imshow('Frame', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
"""

