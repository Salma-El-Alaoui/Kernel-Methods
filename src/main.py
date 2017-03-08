

from HistogramOrientedGradient import HistogramOrientedGradient
from equalization import equalize_item
from data_utils import load_data, train_test_split, write_submission
from image_utils import vec_to_img
from svm import OneVsOneSVM, grid_search_ovo
from kernels import rbf_kernel
import numpy as np


# TODO: shouldn't be here, but somewhere related to hog
# TODO: add capacity to store features
def load_hog_features(rgb=False,equalize=True,n_cells_hog=8):
    data_train, data_test, y_train = load_data()
    hist_train = []
    hog = HistogramOrientedGradient(n_cells=n_cells_hog,cell_size=int(32./n_cells_hog))
    for id_img in range(len(data_train)):
        image = data_train[id_img]
        if equalize:
            img = equalize_item(image, rgb=rgb, verbose=False)
        else:
            img = vec_to_img(image,rgb=rgb)
        hist_train.append(hog._build_histogram(img))
    hist_test = []
    for id_img in range(len(data_test)):
        image = data_test[id_img]
        if equalize:
            img = equalize_item(image, rgb=rgb,verbose=False)
        else:
            img = vec_to_img(image,rgb=rgb)
        hist_test.append(hog._build_histogram(img))
    X_train = np.array(hist_train)
    X_test = np.array(hist_test)
    print("\thog features loaded")
    return X_train, X_test, y_train


#TODO: encapsulate all of the following in a function to be put in main

# define some flags
equalize = False
rgb = False # whether or not to consider 3 different channels (if false, mean of 3 channels)
n_cells_hog = 8

kernel = rbf_kernel # or any other kernel from the kernels.py file
classifier = "one_vs_one"

cross_validation = False
dict_param = {'kernel_param': [1, 2, 3], 'C': [100, 150, 200]}
nb_folds = 5

train_test_val = True
pr_train = 0.8

make_submission = True
submission_name = "test" # suffix to submission file

print("Loading Features ...")
X_train, X_test, y_train = load_hog_features(rgb=rgb,equalize=equalize,n_cells_hog=n_cells_hog)

if cross_validation:
    if classifier == "one_vs_one":
        print("Performing grid search...")
        parameters_dic, best_parameter = grid_search_ovo(X_train=X_train, y_train=y_train, dict_param=dict_param,
                                                         nb_folds=nb_folds, kernel=kernel)
    elif classifier == "crammer_singer":
        #TODO
        pass

if train_test_val:
    print("Splitting into train and validation datasets ...")
    X_train_t, X_train_v, y_train_t, y_train_v = train_test_split(X_train, y_train, pr_train)
    if classifier == "one_vs_one":
        clf = OneVsOneSVM(C=100, kernel=kernel, kernel_param=3)
        print("Fitting classifier...")
        clf.fit(X_train_t, y_train_t)
        score = clf.score(X_train_v, y_train_v)
        print("Accuracy score on validation dataset: ", score)
    elif classifier == "crammer_singer":
        #TODO
        pass

if make_submission:
    if classifier == "one_vs_one":
        clf = OneVsOneSVM(C=100, kernel=kernel, kernel_param=3)
        print("Fitting classifier on all training data...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Writing submission...")
        write_submission(y_pred, submission_name)
    elif classifier == "crammer_singer":
        #TODO
        pass



