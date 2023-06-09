from processing import *
from collections import Counter
import SIFT_transform
import query
import random

folder_path = 'resources/videos/'
db_path = 'resources/db/'
db_name = 'database'
q = query.Query()

print('START')

score = 0
acc = 0

dictionary_locations = {
    "nk": "Nieuwe Kerk",
    "xx": "Unknown",
    "oj": "Oude Jan",
    "rh": "Stadhuis Delft"
}

frequency = 10


def process(results, percent_sift):
    # get a list of tuples, where first is value of sift, second of colorhist
    arr = np.array(results)
    n_rows, n_cols = arr.shape
    counts = []
    for i in range(n_cols):
        counter = Counter(arr[:, i])
        most_common, _ = counter.most_common(1)[0]
        weight = percent_sift if i == 0 else (1-percent_sift)
        counts.append((most_common, weight))

    # combine the counts into one final result
    combined_count = Counter()
    for value, weight in counts:
        combined_count[value] += weight

    # return the most common element from the combined count
    return combined_count.most_common(1)[0][0]


def run(video):
    # for file in os.listdir(folder_path):
    counter = 0
    results = []
    out_frame = None

    #file_path = os.path.join(folder_path, file)

    #print(file_path)
    #cap = cv2.VideoCapture(file_path)
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print("Error opening video file")

    rand_frame = random.randint(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // frequency))

    while cap.isOpened():
        # Read a frame from the video file
        ret, frame = cap.read()

        # Check if a frame was successfully read
        if not ret:
            break

        if counter == 0:
            out_frame = frame

        if abs(counter-rand_frame) == 1:
            out_frame = frame

        # Increment the frame count
        counter += 1
        # Skip frames if the frequency is not met
        if counter % frequency != 0:
            continue

        # Process the frame

        img_resize = cv2.resize(frame, (640, 480))
        keypoints, descriptors = SIFT_transform.make_sift(img_resize)
        chans = cv2.split(img_resize)
        color_hist = np.zeros((256, len(chans)))
        for i in range(len(chans)):
            color_hist[:, i] = np.histogram(chans[i], bins=np.arange(256 + 1))[0] / float(
                (chans[i].shape[0] * chans[i].shape[1]))

        sift_winners, sift_distances, hist_winners, hist_distances = q.find(descriptors, color_hist)
        # print(sift_winners, sift_distances, hist_winners, hist_distances)
        # sift_winners[0].split("_")[2]
        sift_loc = dictionary_locations[sift_winners[0].split("_")[2]]
        hist_loc = dictionary_locations[hist_winners[0].split("_")[2]]

        results.append((sift_loc, hist_loc))

        # Display the frame (optional)
        cv2.imshow('Frame', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("results for file ", results)
    processed = process(results, 0.7)
    # process results
    return processed, out_frame
