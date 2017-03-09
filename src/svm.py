# -*- coding: utf-8 -*-

"""
Created on Wed Feb 15 14:38:54 2017
@author: salma
Binary SVM and One vs One Multiclass SVM.
"""

import numpy as np
import cvxopt
from data_utils import datasets, cross_validation
from kernels import rbf_kernel, linear_kernel
from sklearn.datasets import load_iris
import operator

class BinarySVM:

    def __init__(self, C, kernel, kernel_param):
        self.kernel = kernel
        self.C = C
        self. kernel_param = kernel_param
        self.w = None
        self.b = None
        self.X = None
        self.mu_support = None
        self.idx_support = None
        self.y = None
        self.class_1 = None
        self.class_2 = None

    def _qp(self, H, e, A, b, C=np.inf, l=1e-8, verbose=False):
        # Gram matrix
        n = H.shape[0]
        H = cvxopt.matrix(H)
        A = cvxopt.matrix(A, (1, n))
        e = cvxopt.matrix(-e)
        b = cvxopt.matrix(0.0)
        if C == np.inf:
            G = cvxopt.matrix(np.diag(np.ones(n) * -1))
            h = cvxopt.matrix(np.zeros(n))
        else:
            G = cvxopt.matrix(np.concatenate([np.diag(np.ones(n) * -1),
                                              np.diag(np.ones(n))], axis=0))
            h = cvxopt.matrix(np.concatenate([np.zeros(n), C * np.ones(n)]))

        # Solve QP problem
        cvxopt.solvers.options['show_progress'] = verbose
        solution = cvxopt.solvers.qp(H, e, G, h, A, b)

        # Lagrange multipliers
        mu = np.ravel(solution['x'])
        return mu

    def _svm_solver(self, K, y, C=np.inf):
        H = (y * K).T * y
        e = np.ones(y.shape[0])
        A = y
        b = 0
        mu = self._qp(H, e, A, b, C,  l=1e-8, verbose=False)
        idx_support = np.where(np.abs(mu) > 1e-5)[0]
        mu_support = mu[idx_support]
        return mu_support, idx_support

    def _compute_b(self, K, y, mu_support, idx_support, C):
        mu = np.zeros(K.shape[0])
        mu[idx_support] = mu_support
        # we choose a support vector for which \xi_i = 0, which means \mu_i < C
        i = idx_support[np.where(C - mu_support > 1e-5)[0][0]]
        y_i = y[i]
        b = y_i - np.sum((y[idx_support] * mu_support) * K[i, idx_support], axis=0)
        return b

    def kernel_matrix(self, X):
        return self.kernel(X, X, self.kernel_param)

    def fit(self, X, y, K):

        self.class_1 = np.min(y)
        self.class_2 = np.max(y)
        ind_1 = (y == self.class_1)
        ind_2 = (y == self.class_2)
        new_y = np.zeros(X.shape[0])
        new_y[ind_1] = 1
        new_y[ind_2] = -1

        mu_support, idx_support = self._svm_solver(K, new_y, self.C)
        b = self._compute_b(K, new_y, mu_support, idx_support, self.C)
        w = np.sum((mu_support * new_y[idx_support])[:, None] * X[idx_support], axis=0)
        self.X = X
        self.b = b
        self.w = w
        self.mu_support = mu_support
        self.idx_support = idx_support
        self.y = new_y
        return self

    def predict(self, X_test):
        X_support = self.X[self.idx_support]
        G = self.kernel(X_test, X_support, self.kernel_param)
        # Calcul de la fonction de décision
        decision = G.dot(self.mu_support * self.y[self.idx_support]) + self.b
        y_pred = np.sign(decision)
        y_pred_real = np.zeros(X_test.shape[0], dtype=np.int32)
        for i, y in enumerate(y_pred):
            if y == 1:
                y_pred_real[i] = self.class_1
            else:
                y_pred_real[i] = self.class_2
        return y_pred_real

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class OneVsOneSVM:

    def __init__(self, C, kernel, kernel_param):
        self. C = C
        self.kernel = kernel
        self.kernel_param = kernel_param
        self.clf_matrix = []
        self.n_classes = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        K = self.kernel(X, X, self.kernel_param)

        for i in range(self.n_classes):
            clf_list = []
            for j in range(i + 1, self.n_classes):
                clf_list.append(BinarySVM(C=self.C, kernel=self.kernel, kernel_param=self.kernel_param))
            self.clf_matrix.append(clf_list)

        class_indices = []
        for i in range(self.n_classes):
            class_indices.append((y == i))
        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                ind = np.logical_or(class_indices[i], class_indices[j])
                k = K[ind, :]
                k = k[:, ind]
                self.clf_matrix[i][j - i - 1].fit(X=X[ind, :], y=y[ind], K=k)
        return self

    def predict(self, X_test):
        votes = np.zeros((X_test.shape[0], self.n_classes))
        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                y = self.clf_matrix[i][j - i - 1].predict(X_test)
                for k, pred in enumerate(y):
                    votes[k][pred] += 1
        return np.argmax(votes, axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def grid_search_ovo(X_train, y_train, dict_param, nb_folds, kernel, kernel_param=None, verbose=True):
    parameters = dict()
    if 'kernel_param' in dict_param:
        for C in dict_param['C']:
            for kernel_param in dict_param['kernel_param']:
                accuracies_folds = list()
                for X_train_train, y_train_train, X_valid, y_valid in cross_validation(X_train, y_train, nb_folds):
                    clf = OneVsOneSVM(C=C, kernel=kernel, kernel_param=kernel_param)
                    clf.fit(X_train_train, y_train_train)
                    acc = clf.score(X_valid, y_valid)
                    accuracies_folds.append(acc)
                if verbose:
                    print("\tC = ", C, "kernel param = ", kernel_param, "---- score = ", np.mean(accuracies_folds))
                parameters[("C", C, "kernel_param", + kernel_param)] = np.mean(accuracies_folds)
    else:
        for C in dict_param['C']:
            accuracies_folds = list()
            for X_train_train, y_train_train, X_valid, y_valid in cross_validation(X_train, y_train, nb_folds):
                clf = OneVsOneSVM(C=C, kernel=kernel, kernel_param=kernel_param)
                clf.fit(X_train_train, y_train_train)
                acc = clf.score(X_valid, y_valid)
                accuracies_folds.append(acc)
            if verbose:
                print("\tC = ", C, "kernel parameter = ", kernel_param, "---- score = ", np.mean(accuracies_folds))
            parameters[("C", C, "kernel_param", + kernel_param)] = np.mean(accuracies_folds)

    best_param = max(parameters.items(), key=operator.itemgetter(1))[0]
    if verbose:
        print("\tThe best set of parameters is: ", best_param)
    return parameters, best_param


def unit_test():
    dict_param = {'kernel_param': [1, 2, 3], 'C': [1, 2, 3]}

    X, y = datasets(name='clowns', n_points=200, sigma=0.7)
    clf = BinarySVM(C=np.inf, kernel=rbf_kernel, kernel_param=3.)
    K = clf.kernel_matrix(X)
    clf.fit(X, y, K)
    print("training score on toy dataset", clf.score(X, y))

    iris = load_iris()
    X = iris.data
    Y = iris.target
    _, best = grid_search_ovo(X, Y, dict_param, 5, rbf_kernel)
    print("best parameter on iris", best)

if __name__ == '__main__':
    # unit_test()
    pass





