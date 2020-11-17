import json
import shelve
import numpy as np
import scipy as sp
from numpy import ones, zeros
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import time
import globals

TOP_DIR = globals.TOP_DIR

class Data:
    DATA_DIR = TOP_DIR + 'dataset/'
    SHELF_ALL_PATH = DATA_DIR + 'shelf_all_feats'                 # All features
    SHELF_TOP_PATH = DATA_DIR + 'shelf_top_feats'                 # Top features
    X_PATH = DATA_DIR + 'drebin-parrot-v2-down-features-X.json'
    Y_PATH = DATA_DIR + 'drebin-parrot-v2-down-features-Y.json'

    def __init__(self, x_path=X_PATH, y_path=Y_PATH, n_features=-1,
                 read_shelf=True, test_size=0.3, random_state=0):
        """
        Initialises a Data instance by loading and splitting the dataset.
        :param x_path: Path to the set of features
        :param y_path: Path to the set of labels
        :param n_features: Number of features to select, -1 to include all.
        :param read_shelf: Whether to read from a pre-stored shelf.
        :param test_size: Test size for splitting.
        :param random_state: Random seed for splitting.
        """
        self.X_path = x_path
        self.Y_path = y_path
        self.n_features = n_features
        if n_features ==-1:
            self.shelf_path = Data.SHELF_ALL_PATH
        else:
            self.shelf_path = Data.SHELF_TOP_PATH

        # Load the data
        self.load_data(read_shelf)
        # Split the data
        self.split(test_size=test_size, random_state=random_state)

    def load_data(self, read_shelf):
        """
        Reads data, either from a pre-stored shelf or from the original json
        files.
        :return: None
        """
        if read_shelf:
            try:
                # Attempt reading pre-shelved objects first
                self.__read_shelf()
            except Exception as e:
                print(f'Exception while reading the data shelf ({e})')
                # Otherwise, read data from the the json files
                self.__read_json()
        else:
            self.__read_json()

    def __read_shelf(self):
        """
        Reads data as pickled objects from a pre-stored shelf.
        :return: None
        """
        with shelve.open(self.shelf_path) as shelf:
            print('Starting to read the data shelf')
            t1 = time.perf_counter()
            self.X = shelf['X']                 # Note: X is a sparse.csr_matrix
            self.X_malware = shelf['X_malware']
            self.X_goodware = shelf['X_goodware']
            self.Y = shelf['Y']
            self.feature_names = shelf['feature_names']
            t2 = time.perf_counter()
            print(f'Finished reading the data shelf. '
                  f'Elapsed time: {(t2 - t1) / 60.0} minutes')

    def __read_json(self):
        """
        Reads data from the original json files, and stores them in a shelf.
        :return: None
        """
        try:
            with open(self.X_path) as Xfile, open(self.Y_path) as Yfile:
                print('Starting to read the json files')
                t1 = time.perf_counter()
                Ydata = [obj[0] for obj in json.load(Yfile)]
                Xdata = json.load(Xfile)
                # Note: some samples (31) are empty
                # Xdata = [obj for obj in json.load(Xfile) if len(obj)>1]
                for record in Xdata:
                    del record['sha256']        # Remove signature from features
                t2 = time.perf_counter()
                print(f'Finished reading the json files. '
                      f'Elapsed time: {(t2 - t1) / 60.0} minutes')
        except Exception as e:
            print(f'Exception while reading json files ({e})')
            raise e

        self.Y = np.array(Ydata, dtype=np.uint8)
        print('Y.shape', self.Y.shape)
        self.vectorizer = DictVectorizer(sparse=True, dtype=np.uint8)
        self.X = self.vectorizer.fit_transform(Xdata)
        print('X.shape (All):', self.X.shape)
        # If feature selection is applied
        if self.n_features != -1:
            support = SelectKBest(chi2, k=self.n_features).fit(self.X, self.Y)
            self.vectorizer.restrict(support.get_support())
            self.X = support.transform(self.X)
            print('X.shape (Reduced):', self.X.shape)

        self.feature_names = self.vectorizer.feature_names_
        malware_idx = np.where(self.Y == 1)[0]
        self.X_malware = self.X[malware_idx, :]
        goodware_idx = np.where(self.Y == 0)[0]
        self.X_goodware = self.X[goodware_idx, :]


        try:
            with shelve.open(self.shelf_path, 'c') as shelf:
                print('Saving data to a shelf')
                shelf['X'] = self.X
                shelf['X_malware'] = self.X_malware
                shelf['X_goodware'] = self.X_goodware
                shelf['Y'] = self.Y
                shelf['feature_names'] = self.feature_names
                print('Finished saving the shelf')
        except Exception as e:
            print(f'Exception while saving data to a shelf ({e})')

    def __restrict_features_freq(self, min_count=1):
        """
        Selects features that occur more than min_count in the dataset.
        :param min_count: The frequency threshold
        :return: The indices of features satisfying the minimum frequency count
        """
        col_idx = self.X.tocsc().nonzero()[1]
        counter = np.bincount(col_idx)
        print("Counter:", len(counter))
        include_cols = np.where(counter > min_count)[0]
        return include_cols

    def split(self, test_size=0.3, random_state=0):
        # Split malware
        self.X_mal_train, self.X_mal_test = train_test_split(self.X_malware,
                                                   test_size=test_size,
                                                   random_state=random_state)
        self.Y_mal_train = ones(self.X_mal_train.shape[0])
        self.Y_mal_test = ones(self.X_mal_test.shape[0])


        # Split goodware
        self.X_good_train, self.X_good_test = train_test_split(self.X_goodware,
                                                     test_size=test_size,
                                                     random_state=random_state)
        self.Y_good_train = zeros(self.X_good_train.shape[0])
        self.Y_good_test = zeros(self.X_good_test.shape[0])

        # Combine to construct X_train and Y_train
        self.X_train = sp.sparse.vstack([self.X_mal_train, self.X_good_train])
        self.Y_train = np.concatenate([self.Y_mal_train, self.Y_good_train])

        self.X_test = sp.sparse.vstack([self.X_mal_test, self.X_good_test])
        self.Y_test = np.concatenate([self.Y_mal_test, self.Y_good_test])

        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def get_X(self):
        return self.X

    def get_Y(self):
        return self.Y

    def get_X_malware(self):
        return self.X_malware

    def get_X_goodware(self):
        return self.X_goodware

    def get_feature_names(self):
        return self.feature_names

    def get_X_mal_train(self):
        return self.X_mal_train

    def get_X_mal_test(self):
        return self.X_mal_test

    def get_X_good_train(self):
        return self.X_good_train

    def get_X_good_test(self):
        return self.X_good_test
