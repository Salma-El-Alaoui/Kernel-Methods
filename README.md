# Kernel-Methods
Project for the course Kernel Methods for Machine Learning

## Useful links

- Notebook with CIFAR-10 dataset: https://houxianxu.github.io/implementation/SVM.html

## TODO

*** Friday - Score purpose ***
Features
- BW vs. RGB (with/without equalise)
- RGB: do we need dimension reduction (Kernel PCA)
- cell_size for HoG
- signed/unsigned for HoG
- implement normalisation  for HoG?
- try with input resized to 64*64

SVM
- kernels
- linear
- gaussian
- laplacian
- histogram intersection
- …

*** Sunday - Report ***
- compare one vs one / crammer singer


## TODO

- (opt.) More precise maxima detection using Taylor expansion
- Comment what is done in histogram equalization 
- !!! Enlever l'import de rbf_kernel dans crammer singer et implémenter le notre



- SIFT
    - 1) renvoyer séparément les maxima et les minima pour les 2 comparaisons de 3 DoGs (ie. on saura si les extrema ont été obtenus pour \sigma ou pour k*\sigma)
        >> cf. http://aishack.in/tutorials/sift-scale-invariant-feature-transform-keypoints/ 
        >> cf. dessin de Juliette :)

    - 2) il faudra trouver les bad points (corners, edges, low contrast) pour les deux DoGs concernés (les DoGs du milieu)

## Notes

- Bag of features
    - tf_doc(word) = n_occurences_in_doc(word)/n_words_in_doc
    - idf(word) = log(1 + n_docs/n_docs_that_contains(word))

    - TODO: see if we want to change tf (cf. wikipedia tf-idf), for example with a log scale. Indeed, there are different options for tf-idf


