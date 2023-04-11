from processing import *
from matching import *
import SIFT_transform
import query


def get_key(path):
    loc_code = path.split('_')[2]
    loc = path.split('_')[5].split('.')[0] if len(path.split('_')) > 5 else path.split('_')[4].split('.')[0]
    return loc_code + '_' + loc


def get_predicted_location(video_path):
    # video_name = 'VIDEO0191'
    # video_path = 'resources/videos/' + video_name + '.avi'

    frames = get_frames(video_path, 10)
    poss_loc = {}
    for i in range(len(frames)):
        frame = frames[i]
        keypoints, descriptors = SIFT_transform.make_sift(frame)

        chans = cv2.split(frame)
        color_hist = np.zeros((256, len(chans)))
        for i in range(len(chans)):
            color_hist[:, i] = np.histogram(chans[i], bins=np.arange(256 + 1))[0] / float(
                (chans[i].shape[0] * chans[i].shape[1]))

        sift_winners, sift_distances, hist_winners, hist_distances = q.find(descriptors, color_hist)

        for sw in sift_winners:
            key = get_key(sw)
            poss_loc[key] = poss_loc[key] + 1 if key in poss_loc else 1

        for hw in hist_winners:
            key = get_key(hw)
            poss_loc[key] = poss_loc[key] + 1 if key in poss_loc else 1

    return sorted(poss_loc)[0].split('_')[0]


folder_path = 'resources/videos/'
db_path = 'resources/db/'
db_name = 'database'

print('START')

q = query.Query()

gt_file = open('resources/ground_truth.txt', 'r')
lines = gt_file.readlines()
gt = {}
for line in lines:
    video, location = line.strip().split(' ')
    gt[video] = location

accuracy = 0
total = 0
for file in os.listdir(folder_path):
    total += 1
    file_path = os.path.join(folder_path, file)
    print(file_path)
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("Error opening video file")

    video_name = file_path.split('/')[2].split('.')[0]
    predicted_location = get_predicted_location(file_path)

    if gt[video_name] == predicted_location:
        print('CORRECT')
        accuracy += 1
    else:
        print('WRONG')
    # while cap.isOpened():
    #     # Read a frame from the video file
    #     ret, frame = cap.read()
    #
    #     # Check if a frame was successfully read
    #     if not ret:
    #         break
    #
    #     # Process the frame
    #     img_resize = cv2.resize(frame, (640, 480))
    #     keypoints, descriptors = SIFT_transform.make_sift(img_resize)
    #     chans = cv2.split(img_resize)
    #     color_hist = np.zeros((256, len(chans)))
    #     for i in range(len(chans)):
    #         color_hist[:, i] = np.histogram(chans[i], bins=np.arange(256 + 1))[0] / float(
    #             (chans[i].shape[0] * chans[i].shape[1]))
    #
    #     sift_winners, sift_distances, hist_winners, hist_distances = q.find(descriptors, color_hist)
    #     print(sift_winners, sift_distances, hist_winners, hist_distances)
    #
    #     # Display the frame (optional)
    #     cv2.imshow('Frame', frame)
    #
    #     # Exit the loop if the 'q' key is pressed
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         break

    cap.release()
    cv2.destroyAllWindows()

print('Overall accuracy: ', accuracy / total)
print('END')

