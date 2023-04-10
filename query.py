import pickle
import image_search

base = 'resources/db/'
db_name = 'resources/db/database.sqlite'


class Query:
    def __init__(self, db):
        search = image_search.Searcher(db_name)  # odakle mi searcher

    def find(sift_query, hist_query, candidates=3):
        sift_winners = None
        sift_distances = None
        hist_winners = None
        hist_distances = None
        # query sift
        fnames = base + 'sift_vocabulary.pkl'
        with open(fnames, 'rb') as f:
            sift_vocabulary = pickle.load(f, encoding="bytes")

        # Get a histogram of visual words for the query image
        image_words_s = sift_vocabulary.project(sift_query)
        sift_candidates = search.query_iw('sift', image_words_s)

        # query colorhist
        fnameh = base + 'hist_vocabulary.pkl'
        with open(fnameh, 'rb') as f:
            colorhist_vocabulary = pickle.load(f, encoding="bytes")

        image_words_h = colorhist_vocabulary.project(hist_query)
        colorhist_candidates = search.query_iw('colorhist', image_words_h)

        if sift_candidates is not None:
            sift_winners = [search.get_filename(cand[1]) for cand in sift_candidates][0:candidates]
            sift_distances = [cand[0] for cand in sift_candidates][0:candidates]

        if colorhist_candidates is not None:
            hist_winners = [search.get_filename(cand[1]) for cand in colorhist_candidates][0:candidates]
            hist_distances = [cand[0] for cand in colorhist_candidates][0:candidates]

        return sift_winners, sift_distances, hist_winners, hist_distances
