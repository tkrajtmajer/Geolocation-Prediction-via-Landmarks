import numpy as np
import sqlite3
import pickle

# bag of words
# index feature descriptors to query only nearest neighbours, not entire db


class Indexer(object):
    def __init__(self, db):
        self.con = sqlite3.connect(db)

    def __del__(self):
        self.con.close()

    def db_commit(self):
        self.con.commit()

    def create_tables(self):
        self.con.execute('CREATE TABLE imlist(filename)')
        self.con.execute('CREATE INDEX im_idx ON imlist(filename)')

        self.con.execute('CREATE TABLE sift_imwords(imid,wordid,vocname)')
        self.con.execute('CREATE INDEX sift_imid_idx ON sift_imwords(imid)')
        self.con.execute('CREATE INDEX sift_wordid_idx ON sift_imwords(wordid)')

        self.con.execute('CREATE TABLE sift_imhistograms(imid,histogram,vocname)')
        self.con.execute('CREATE INDEX sift_imidhist_idx ON sift_imhistograms(imid)')

        self.con.execute('CREATE TABLE colorhist_imwords(imid,wordid,vocname)')
        self.con.execute('CREATE INDEX colorhist_imid_idx ON colorhist_imwords(imid)')
        self.con.execute('CREATE INDEX colorhist_wordid_idx ON colorhist_imwords(wordid)')

        self.con.execute('CREATE TABLE colorhist_imhistograms(imid,histogram,vocname)')
        self.con.execute('CREATE INDEX colorhist_imidhist_idx ON colorhist_imhistograms(imid)')

        self.con.execute('CREATE TABLE combined_imwords(imid,wordid,vocname)')
        self.con.execute('CREATE INDEX combined_imid_idx ON combined_imwords(imid)')
        self.con.execute('CREATE INDEX combined_wordid_idx ON combined_imwords(wordid)')

        self.con.execute('CREATE TABLE combined_imhistograms(imid,histogram,vocname)')
        self.con.execute('CREATE INDEX combined_imidhist_idx ON combined_imhistograms(imid)')

        self.db_commit()


    def add_index(self, image_name, features, voc):
        image_id = self.get_id(image_name)
        image_words = voc.project(features)
        for i in range(np.shape(image_words)[0]):
            word = image_words[i]
            self.con.execute("INSERT INTO combined_imwords(imid,wordid,vocname) VALUES (?,?,?)",
                             (image_id, word, voc.name))
        self.con.execute("INSERT INTO combined_imhistograms(imid,histogram,vocname) VALUES (?,?,?)", (image_id, pickle.dumps(image_words), voc.name))


    def add_index_SIFT(self, image_name, descriptor, voc):
        image_id = self.get_id(image_name)

        image_words = voc.project(descriptor)

        for i in range(np.shape(image_words)[0]):
            word = image_words[i]
            self.con.execute("INSERT INTO sift_imwords(imid,wordid,vocname) VALUES (?,?,?)",
                             (image_id, word, voc.name))

        self.con.execute("INSERT INTO sift_imhistograms(imid,histogram,vocname) VALUES (?,?,?)", (image_id, pickle.dumps(image_words), voc.name))

    def add_index_colorhist(self, image_name, descriptor, voc):
        image_id = self.get_id(image_name)

        image_words = voc.project(descriptor)

        for i in range(np.shape(image_words)[0]):
            word = image_words[i]
            self.con.execute("INSERT INTO colorhist_imwords(imid,wordid,vocname) VALUES (?,?,?)",
                             (image_id, word, voc.name))

        self.con.execute("INSERT INTO colorhist_imhistograms(imid,histogram,vocname) VALUES (?,?,?)",
                         (image_id, pickle.dumps(image_words), voc.name))

    def get_id(self, image_name):
        res = self.con.execute("SELECT rowid FROM imlist WHERE filename='%s'" % image_name).fetchone()

        if res is None:
            cur = self.con.execute("INSERT INTO imlist(filename) VALUES ('%s')" % image_name)
            return cur.lastrowid

        return res[0]

    def is_indexed(self, image_name):
        image = self.con.execute("SELECT rowid FROM imlist WHERE filename='%s'" % image_name).fetchone()
        return image is not None
