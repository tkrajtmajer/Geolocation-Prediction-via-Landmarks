# Makes database file from images provided in resources/images folder

import sys
import os
import cv2
import sqlite3
import numpy as np
import SIFT_transform

folder_path = 'resources/images/'

# create database
db_path = 'resources/db/'
db_name = 'database'

# check if database already exists
new = False
if os.path.isfile(db_path + db_name + '.sqlite'):
    action = input('Database already exists. Do you want to (r)emove, (a)ppend or (q)uit? ')
    print('action =', action)
else:
    action = 'c'
if action == 'r':
    print('removing database', db_name, '...')
    os.remove(db_path + db_name + '.sqlite')
    new = True
elif action == 'a':
    print('appending to database ... ')
elif action == 'c':
    print('creating database', db_name, '...')
    new = True
else:
    print('Quit database tool')
    sys.exit(0)

if new:
    # create tables
    connection = sqlite3.connect(db_path + db_name + '.sqlite')
    c = connection.cursor()

    c.execute('''CREATE TABLE images
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              image_path TEXT,
              keypoint BLOB,
              descriptor BLOB)''')

    for folder in os.listdir(folder_path):
        full_path = os.path.join(folder_path, folder)
        if os.path.isdir(full_path):
            print("reading folder " + folder)

            for file in os.listdir(full_path):

                if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                    file_path = os.path.join(full_path, file)

                    img = cv2.imread(file_path)
                    img_resize = cv2.resize(img, (640, 480))

                    # DEBUG display image
                    '''
                    cv2.imshow("imaz", img_resize)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    '''

                    # compute sift for image
                    keypoints, descriptors = SIFT_transform.make_sift(img_resize)

                    # DEBUG display SIFT
                    '''
                    img_with_SIFT = cv2.drawKeypoints(img_resize, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv2.imshow("imaz w keypoints", img_with_SIFT)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    '''

                    # save image descriptors
                    keypoint_bytes = np.array([kp.pt for kp in keypoints]).tobytes()
                    descriptor_bytes = descriptors.tobytes()

                    c.execute("INSERT INTO images (image_path, keypoint, descriptor) VALUES (?, ?, ?)", (file_path, keypoint_bytes, descriptor_bytes))
                    connection.commit()

    connection.close()
