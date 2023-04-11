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
vocab_path_sift = 'resources/db/sift_vocabulary.pkl'
vocab_path_hist = 'resources/db/hist_vocabulary.pkl'
num_clusters = 50

# check if database already exists
new = False
if os.path.isfile(db_path + db_name + '.sqlite'):
    action = input('Database already exists. Do you want to (m)ake new, (s)kip or (q)uit? ')
    print('action =', action)
else:
    action = 'c'
if action == 'm':
    print('creating database', db_name, '...')
    os.remove(db_path + db_name + '.sqlite')
    new = True
elif action == 's':
    print('skipping ... ')
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
    features_SIFT = {}
    features_colorhist = {}
    features_combined = {}
    image_list = np.array([])
    vocabularySIFT = None
    vocabularyHist = None

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
                    features_SIFT[file_path] = descriptors
                    #print(descriptors.shape)

                    # compute colorhist for image
                    chans = cv2.split(img_resize)
                    color_hist = np.zeros((256, len(chans)))
                    for i in range(len(chans)):
                        color_hist[:, i] = np.histogram(chans[i], bins=np.arange(256 + 1))[0] / float((chans[i].shape[0] * chans[i].shape[1]))
                    features_colorhist[file_path] = color_hist
                    #print(color_hist.flatten().shape)

                    #features_combined[file_path] = np.concatenate((descriptors.flatten(), color_hist.flatten()), axis=0)
                    #feature_vector = descriptors

                    #features_combined[file_path] =

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
        pickle.dump(features_SIFT, f)

    # Create a visual vocabulary (Bag of Words) from the sift extracted features.
    if os.path.isfile(vocab_path_sift):
        compute = input("Found existing vocabulary: " + vocab_path_sift + " Do you want to recompute it? ([Y]/N): ")
    else:
        compute = 'Y'
    """
    if compute == 'Y' or compute == '':
        print('Creating SIFT vocabulary ... ')
        sift_vocabulary = clustering.Cluster("cluster")
        sift_vocabulary.train(features_SIFT, num_clusters)
        with open(vocab_path, 'wb') as f:
            pickle.dump(sift_vocabulary, f)

    print('\nAdding sift features to database ...\n')
    for i in range(len(image_list)):
        indexer.add_index(image_list[i], features_SIFT[image_list[i]], sift_vocabulary)
    """

    if compute == 'Y' or compute == '':
        print('Creating combined vocabulary ... ')
        vocabularySIFT = clustering.Cluster("sift cluster")
        vocabularySIFT.train(features_SIFT, num_clusters)
        with open(vocab_path_sift, 'wb') as f:
            pickle.dump(vocabularySIFT, f)

        vocabularyHist = clustering.Cluster("hist cluster")
        vocabularyHist.train(features_colorhist, num_clusters)
        with open(vocab_path_hist, 'wb') as f:
            pickle.dump(vocabularyHist, f)

    print('\nAdding features to database ...\n')
    for i in range(len(image_list)):
        indexer.add_index_SIFT(image_list[i], features_SIFT[image_list[i]], vocabularySIFT)
        indexer.add_index_colorhist(image_list[i], features_colorhist[image_list[i]], vocabularyHist)

    indexer.db_commit()
    print('\nDone\n')
