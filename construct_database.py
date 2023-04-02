# Makes database file from images provided in resources/images folder

import sys
import os
import cv2
import sqlite3
import numpy as np
import pickle
import SIFT_transform
import indexer
import clustering


folder_path = 'resources/images/'

# create database
db_path = 'resources/db/'
db_name = 'database'

features_path = 'resources/db/sift_features.pkl'
vocab_path = 'resources/db/sift_vocabulary.pkl'
num_clusters = 50

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
    # create indexer
    indexer = indexer.Indexer(db_path + db_name + '.sqlite')
    indexer.create_tables()
    # store SIFT computed for each image in dictionary
    features = {}
    image_list = np.array([])
    sift_vocabulary = None

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
                    features[file_path] = descriptors
                    image_list = np.append(image_list, file_path)

                    # DEBUG display SIFT
                    '''
                    img_with_SIFT = cv2.drawKeypoints(img_resize, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv2.imshow("imaz w keypoints", img_with_SIFT)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    '''

    # compute SIFT features and vocabulary
    with open(features_path, 'wb') as f:
        print('saving features to', features_path, '...')
        pickle.dump(features, f)

    # Create a visual vocabulary (Bag of Words) from the sift extracted features.
    if os.path.isfile(vocab_path):
        compute = input("Found existing vocabulary: " + vocab_path + " Do you want to recompute it? ([Y]/N): ")
    else:
        compute = 'Y'
    if compute == 'Y' or compute == '':
        print('Creating SIFT vocabulary ... ')
        sift_vocabulary = clustering.Cluster("cluster")
        sift_vocabulary.train(features, num_clusters)
        with open(vocab_path, 'wb') as f:
            pickle.dump(sift_vocabulary, f)

    print('\nAdding sift features to database ...\n')
    for i in range(len(image_list)):
        indexer.add_index(image_list[i], features[image_list[i]], sift_vocabulary)

    indexer.db_commit()
    print('\nDone\n')
