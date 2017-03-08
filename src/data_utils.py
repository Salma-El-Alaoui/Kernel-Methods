import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def datasets(name, n_points, sigma=None):
    """Generate artificial datasets
    Parameters
    ----------
    name : str
        The name of dataset. Can be "mixture", "gaussian", "checkers" or "clowns".
    n_points : int
        The number of simulated points.
    sigma : float
        The standard deviation of the noise.
    Returns
    -------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,)
        The targets. 1 or -1 for each sample.
    Notes
    -----
    'mixture'  : 2 classes
    First class is a a mixture of gaussian
    which takes in sandwiches the second class
    'gaussian' : is a classical dataset with two gaussianly distributed
    classes with similar variances but different means
    'checkers': Checkers
    'clowns' : The Clowns
    """
    X = []
    y = []

    if name == 'mixture':
        if sigma is None:
            sigma = 1

        n_points = n_points // 3
        x1 = sigma * np.random.randn(n_points) + 0.3
        x2 = sigma * np.random.randn(n_points) - 0.3
        x3 = sigma * np.random.randn(n_points) - 1
        y1 = sigma * np.random.randn(n_points) + 0.5
        y2 = sigma * np.random.randn(n_points) - 0.5
        y3 = sigma * np.random.randn(n_points) - 1
        X = np.c_[np.r_[x1, x2, x3], np.r_[y1, y2, y3]]
        y = np.concatenate([np.ones(n_points), -np.ones(n_points), np.ones(n_points)])
    elif name == 'gaussian':
        n_points = n_points // 2
        if sigma is None:
            sigma = 1.5
        x1 = sigma * np.random.randn(n_points, 2) + [2, 1]
        x2 = sigma * np.random.randn(n_points, 2) + [-2, -2]
        X = np.r_[x1, x2]
        y = np.concatenate([np.ones(n_points), -np.ones(n_points)])
    elif name == 'checkers':
        nb = n_points // 16
        for i in range(-2, 2):
            for j in range(-2, 2):
                this_X = np.c_[i + np.random.rand(nb), j + np.random.rand(nb)]
                if len(X) == 0:
                    X = this_X
                else:
                    X = np.r_[X, this_X]
                y = np.r_[y, (2 * ((i + j + 4) % 2) - 1) * np.ones(nb)]
    elif name == 'clowns':
        if sigma is None:
            sigma = 0.1

        n_points = n_points // 2
        x1 = 6 * np.random.rand(n_points, 1) - 3
        x2 = x1 ** 2 + np.random.randn(n_points, 1)
        x0 = sigma * np.random.randn(n_points, 2)
        x0 += np.ones((n_points, 1)).dot([[0, 6]])
        X = np.r_[x0, np.c_[x1, x2]]
        X -= np.mean(X, axis=0)[np.newaxis, :]
        X /= np.std(X, axis=0)[np.newaxis, :]
        y = np.r_[np.ones(n_points), -np.ones(n_points)]
    return X, y


def plot_dataset(X, y):
    plt.plot(X[y > 0, 0], X[y > 0, 1], 'rx')
    plt.plot(X[y < 0, 0], X[y < 0, 1], 'bx')


def cross_validation(X, y, nb_folds):
    subset_size = int(len(X) / nb_folds)
    for k in range(nb_folds):
        X_train = np.concatenate((X[:k * subset_size], X[(k + 1) * subset_size:]), axis=0)
        X_test = X[k * subset_size:][:subset_size]
        y_train =  np.concatenate((y[:k * subset_size], y[(k + 1) * subset_size:]), axis=0)
        y_test = y[k * subset_size:][:subset_size]
        yield X_train, y_train, X_test, y_test


def load_data():
    X_train = np.genfromtxt('../data/Xtr.csv', delimiter=',')
    print("\tX_train loaded")
    y_train = np.genfromtxt('../data/Ytr.csv', delimiter=',')
    print("\ty_train loaded")
    X_test = np.genfromtxt('../data/Xte.csv', delimiter=',')
    print("\tX_test loaded")
    X_train = X_train[:,:-1]
    X_test = X_test[:,:-1]
    y_train = y_train[1:,1]
    return X_train, X_test, y_train


def write_submission(y_pred, submission_suffix):
    df = pd.DataFrame(y_pred, columns=['Prediction'])
    df.index += 1
    df['Id'] = df.index
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[0:-1]
    df = df[cols]
    file_name = "../submissions/submission_" + submission_suffix + ".csv"
    df.to_csv(file_name, index=False)


def train_test_split(X, y, pr_train):
    n_train = int(pr_train * len(X))
    X_train_t = X[:n_train]
    X_train_v = X[n_train:]
    y_train_t = y[:n_train]
    y_train_v = y[n_train:]
    return X_train_t, X_train_v, y_train_t, y_train_v