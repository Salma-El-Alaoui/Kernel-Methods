# Kernel-Methods
Project for the course Kernel Methods for Machine Learning

# How to reproduce our result (start.py)
Just run the file start.py, in the src folder. 
It will compute the HOG features, perform a Kernel PCA and a SVM (with Kernel),
with the parameters that were selected with cross validation (cf. main.py for further details).

# Classes containing kernel methods (all in src)
- kernels.py: implements the kernels (linear, rbf, ...)
- crammer_singer_svm.py: implements Crammer Singer SVM
- svm.py: implements One-vs-One SVM
- KernelPCA.py: implements Kernel PCA

# Other useful classes (still in src)
- main.py: file that was used to perform all the experiments (cross validation, build submission,...)
- data_utils.py
- image_utils.py
- equalization.py: performs histogram equalization
- HistogramOrientedGradients.py: to compute the HOG descriptors

# Other files
We tried to implement SIFT, but unfortunately, we could not make it work in time (SIFT folder)