import numpy as np
import os
from scipy.cluster.vq import *


class Cluster(object):
    def __init__(self, name):
        self.name = name
        self.voc = []
        self.idf = []
        self.training_data = []
        self.nbr_words = 0

    def train(self, features, k=100, subsampling=10):
        """ Train a vocabulary from a dictionary of features
                using k-means with k number of words. Subsampling
                of training data can be used for speedup. """

        nbr_desc = len(features)
        # stack all features for k-means
        # create empty array, so we can stack new values against it
        descriptors = np.array([], dtype=np.float32).reshape(0, list(features.values())[0].shape[1])
        for feat in list(features.values()):
            descriptors = np.vstack((descriptors, feat))

        # k-means
        self.voc, distortion = kmeans(descriptors[::subsampling, :], k, 1)
        self.nbr_words = self.voc.shape[0]

        # go through all training images and project on vocabulary
        imwords = np.zeros((nbr_desc, self.nbr_words))
        count = 0
        for desc in list(features.values()):
            imwords[count] = self.project(desc)
            count += 1

        nbr_occurences = np.sum((imwords > 0) * 1, axis=0)
        self.idf = np.log((1.0 * nbr_desc) / (1.0 * nbr_occurences + 1))
        self.training_data = features

    def project(self, descriptors):
        """ project descriptors on the vocabulary
                to create a histogram of words"""
        imhist = np.zeros(self.nbr_words)
        words, distance = vq(descriptors, self.voc)
        for w in words:
            imhist[w] += 1
        return imhist
