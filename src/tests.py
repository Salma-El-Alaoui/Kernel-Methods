from equalization import equalize_item
from image_utils import load_data
from HoG import hog
import numpy as np
from sklearn.svm import SVC

X_train, X_test, y_train = load_data()


hist_train = []
for id_img in range(len(X_train)):
    image = X_train[id_img]
    img = equalize_item(image, verbose=False)
    hist_train.append(hog(img, visualise=False))


hist_test = []
for id_img in range(len(X_test)):
    image = X_test[id_img]
    img = equalize_item(image, verbose=False)
    hist_test.append(hog(img, visualise=False))

hist_train_np = np.array(hist_train)
hist_test_np = np.array(hist_test)

X_train = np.zeros((hist_train_np.shape[0], hist_train_np.shape[1] * hist_train_np.shape[2] * hist_train_np.shape[3]))
X_test = np.zeros((hist_test_np.shape[0], hist_test_np.shape[1] * hist_test_np.shape[2] * hist_test_np.shape[3]))

for i in range(hist_train_np.shape[0]):
    X_train[i] = hist_train_np[i].reshape(hist_train_np.shape[1] * hist_train_np.shape[2] * hist_train_np.shape[3])

for i in range(hist_test_np.shape[0]):
    X_test[i] = hist_test_np[i].reshape(hist_test_np.shape[1] * hist_test_np.shape[2] * hist_test_np.shape[3])


X_train_train = X_train[:4000, :]
X_train_val = X_train[4000:, :]

y_train_train = y_train[:4000]
y_train_val = y_train[4000:]


#gammas = [0.01, 1., 100, 10000]
Cs = [0.01, 1., 100, 10000]

for C in Cs:
    print("--- C = ")
    svm = SVC(kernel='linear', C=C)
    svm.fit(X_train_train, y_train_train)
    print(svm.score(X_train_val, y_train_val))

# %%
from sklearn.model_selection import GridSearchCV

parameters = {'C': Cs}
svm = SVC(kernel='linear')

clf = GridSearchCV(svm, parameters, verbose=2, cv=5)
clf.fit(X_train, y_train)
# %%
print(clf.best_score_)
print(clf.best_params_)

