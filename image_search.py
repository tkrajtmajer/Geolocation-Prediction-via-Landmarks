import sqlite3 as sqlite
import pickle
import numpy as np

class Searcher:
    def __init__(self, db):
        self.con = sqlite.connect(db)

    def __del__(self):
        self.con.close()

    