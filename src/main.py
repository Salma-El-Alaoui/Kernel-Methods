from HistogramOrientedGradient import HistogramOrientedGradient
from equalization import equalize_item
from data_utils import load_data, train_test_split, write_submission
from image_utils import vec_to_img,rgb_to_yuv,yuv_to_rgb,rgb_to_opprgb
from svm import OneVsOneSVM, grid_search_ovo
from kernels import simple_wavelet_kernel, rbf_kernel, laplacian_kernel, linear_kernel
import numpy as np
from KernelPCA import KernelPCA
from crammer_singer_svm import CrammerSingerSVM, grid_search_crammer_singer


# TODO: shouldn't be here, but somewhere related to hog
# TODO: add capacity to store features

def load_hog_features(rgb=False, equalize=True, yuv=False, n_cells_hog=8,signed=False):
    data_train, data_test, y_train = load_data()
    
    hist_train = []
    hog = HistogramOrientedGradient(n_cells=n_cells_hog,cell_size=int(32./n_cells_hog))   

    for id_img in range(len(data_train)):
        image = data_train[id_img]
        if equalize:
            img = equalize_item(image, rgb=rgb, verbose=False)
        elif yuv:
            img = rgb_to_opprgb(image)
        else:
            img = vec_to_img(image, rgb=rgb)
        hist_train.append(hog._build_histogram(img))
    hist_test = []
    
    for id_img in range(len(data_test)):
        image = data_test[id_img]
        if equalize:

            img = equalize_item(image, rgb=rgb,verbose=False)
        elif yuv:
            img = rgb_to_opprgb(image)
        else:
            img = vec_to_img(image, rgb=rgb)
        hist_test.append(hog._build_histogram(img))
    X_train = np.array(hist_train)
    X_test = np.array(hist_test)
    print("\thog features loaded")
    return X_train, X_test, y_train


# TODO: encapsulate all of the following in a function to be put in main

# define some flags
signed = True
equalize = True
rgb = True  # whether or not to consider 3 different channels (if false, mean of 3 channels)
n_cells_hog = 4
yuv = False

kernel = linear_kernel  # or any other kernel from the kernels.py file
kernel_pca =  linear_kernel
classifier = "one_vs_one"
#classifier = "crammer_singer"

cross_validation = True
dict_param = {'kernel': laplacian_kernel, # can't be a list
              'kernel_param': [0.1,0.5,1.,5.],
              'C': [100,1000],
              'apply_pca': True, # can't be a list
              'kernel_pca': linear_kernel, # can't be a list
              'kernel_param_pca': [0], #[0.1,1.,10.],
              'nb_components': [500]}
nb_folds = 5

train_test_val = False
pr_train = 0.8

make_submission = False
submission_name = "test"  # suffix to submission file


load_features = True
path_train_load = "../features/rgb_equalize_4c_train.npy"
path_test_load = "../features/rgb_equalize_4c_test.npy"

save_features = False 
path_train_save = "../features/rgb_notequalize_4c_train"
path_test_save ="../features/rgb_notequalize_4c_test"


if load_features:
    print("Loading Features from file...")
    X_train = np.load(path_train_load)
    X_test = np.load(path_test_load)
    y_train = np.genfromtxt('../data/Ytr.csv', delimiter=',')
    y_train = y_train[1:, 1]
else:
    print("Computing Features...")
    X_train, X_test, y_train = load_hog_features(rgb=rgb, equalize=equalize, n_cells_hog=n_cells_hog)

if cross_validation:
    print("Performing grid search...")
    if classifier == "one_vs_one":
        parameters_dic, best_parameter = grid_search_ovo(X_train=X_train, y_train=y_train, dict_param=dict_param,
                                                         nb_folds=nb_folds)
    elif classifier == "crammer_singer":
        parameters_dic, best_parameter = grid_search_crammer_singer(X_train=X_train, y_train=y_train, dict_param=dict_param,
                                                         nb_folds=nb_folds)

if train_test_val:
    print("Splitting into train and validation datasets...")
    X_train_t, X_train_v, y_train_t, y_train_v = train_test_split(X_train, y_train, pr_train)
    print("Performing KPCA ...")
    kpca = KernelPCA(kernel=kernel_pca, param_kernel=0.65, n_components=800)
    X_train_kpca = kpca.fit_transform(X_train_t)
    X_test_kpca = kpca.transform(X_train_v)
    print("Fitting classifier...")
    if classifier == "one_vs_one":
        clf = OneVsOneSVM(C=1000, kernel=kernel, kernel_param=2)
        # clf.fit(X_train_t, y_train_t)
        # score = clf.score(X_train_v, y_train_v)
    elif classifier == "crammer_singer":
        clf = CrammerSingerSVM(C=0.016, kernel=linear_kernel, param_kernel=None)
    clf.fit(X_train_kpca, y_train)
    score = clf.score(X_test_kpca, y_train_v)
    print("\tAccuracy score on validation dataset: ", score)


if make_submission:
    kpca = KernelPCA(kernel=kernel_pca, param_kernel=1.0, n_components=500)
    X_train_pca = kpca.fit_transform(X_train)
    X_test_pca = kpca.transform(X_test)
    print("Fitting classifier on all training data...")
    if classifier == "one_vs_one":
        clf = OneVsOneSVM(C=100, kernel=kernel, kernel_param=3)
    elif classifier == "crammer_singer":
        clf = CrammerSingerSVM(C=0.016, kernel=linear_kernel, param_kernel=None)
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    print("Writing submission...")
    write_submission(y_pred, submission_name)

if save_features:
    np.save(path_train_save, X_train)
    np.save(path_test_save, X_test)
