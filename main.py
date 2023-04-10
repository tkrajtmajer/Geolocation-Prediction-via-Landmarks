import cv2
import numpy as np
from processing import *
from matching import *
import evaluate
import geolocation
import cgi
import indexer
# import database_reader
print('START')

db_path = 'resources/db/'
db_name = 'database'
# indexer = indexer.Indexer(db_path + db_name + '.sqlite')


def get_name(file):
    print(file)
    names = np.array(file.split("_"))
    return names[-1:][0].split(".")[0]


# database path
database_path = "resources/db.txt"

# read gt values
ground_truth = evaluate.read_ground_truth(True)

# get user video
form = cgi.FieldStorage()
user_video = form.getvalue('file')

# VIDEO0191.avi
# print("Input video name:")
video_name = 'VIDEO0191.avi' # input()

video_path = 'resources/videos/' + video_name
frequency = 10
frames = get_frames(video_path, frequency)
score = 0
acc = 0

for i in range(len(frames)):
    frame = frames[i]
    cont = frame[1]
    # print('main', cont)
    print("Matching frame " + str(i))
    match_list = sift_matching(cont, database_path)

    # get the top prediction and compare with the truth label
    name = ''
    prediction = ''
    # name = get_name(frame[0])
    # prediction = get_name(match_list[0][0])

    input_prediction = geolocation.find_location(frame[0])
    db_prediction = geolocation.find_location(match_list[0][0])

    # if (name == prediction):
    #     acc += 1
    # print(name, pred, "accuracy =", acc / len(frames))

    if input_prediction is not None and db_prediction is not None:
        if np.isclose(input_prediction, db_prediction, rtol=0.1):
            print('Correct guess')
        else:
            print('NOT correct')


"""
# Check if file was uploaded
if user_video.filename:
    # Get the filename
    filename = os.path.basename(user_video.filename)

    # Create a temporary file to store the uploaded video
    with open(filename, 'wb') as tmpfile:
        # Read the uploaded video data and write it to the temporary file
        tmpfile.write(user_video.file.read())

    # Open the temporary file using VideoCapture
    cap = cv2.VideoCapture(filename)

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv2.imshow('Frame',frame)
            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break

    # When everything done, release the video capture object and delete the temporary file
    cap.release()
    os.remove(filename)

# If no file was uploaded, display an error message
else:
    print('No file was uploaded')
"""
