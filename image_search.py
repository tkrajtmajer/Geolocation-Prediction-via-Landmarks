import sqlite3 as sqlite
import pickle
import numpy as np


class Searcher:
    def __init__(self, db):
        self.con = sqlite.connect(db, check_same_thread=False)

    def __del__(self):
        self.con.close()

    def candidates_from_word(self, type, imword):
        """ Get list of images containing imword/ """

        im_ids = self.con.execute(
            "select distinct imid from " + type + "_imwords where wordid=%d" % imword).fetchall()
        return [i[0] for i in im_ids]

    def candidates_from_histogram(self, type, imwords):
        """ Get list of images wth similar words. """

        # get the word ids
        words = imwords.nonzero()[0]
        # find candidates
        candidates = []
        for word in words:
            c = self.candidates_from_word(type, word)
            candidates += c

        # take all unique words and reverse sort on occurence
        tmp = [(w, candidates.count(w)) for w in set(candidates)]
        tmp.sort(key=lambda x: x[1])
        tmp.reverse()

        # return sorted list, best matches first
        return [w[0] for w in tmp]

    def get_imhistogram(self, type, imname):
        """ Return the word histogram for an image. """

        im_id = self.get_imid(imname)
        s = self.con.execute(
            "select histogram from " + type + "_imhistograms where rowid='%d'" % im_id).fetchone()

        # use pickle to decode NumPy arrays from string
        result = None
        try:
            result = pickle.loads(s[0].encode(), encoding="bytes")
        except:
            result = pickle.loads(s[0], encoding="bytes")
        return result

    def query_iw(self, type, h):
        """ Find a list of matching images for image histogram h"""
        candidates = self.candidates_from_histogram(type, h)
        print(candidates)
        matchscores = []
        for imid in candidates:
            # get the name
            cand_name = self.con.execute(
                "SELECT filename FROM imlist WHERE rowid=%d" % imid).fetchone()

            cand_h = self.get_imhistogram(type, cand_name[0])

            cand_dist = np.sqrt(np.sum((h - cand_h) ** 2))  # use L2 distance
            matchscores.append((cand_dist, imid))
        # return a sorted list of distances and databse ids
        matchscores.sort()
        return matchscores

    def get_imid(self, imname):
        im_id = self.con.execute("SELECT rowid FROM imlist WHERE filename='%s'" % imname).fetchone()
        return im_id

    def get_filename(self, imid):
        """ Return the filename for an image id"""
        s = self.con.execute(
            "SELECT filename FROM imlist WHERE rowid='%d'" % imid).fetchone()
        return s[0]
