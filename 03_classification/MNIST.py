#%% 
from lib2to3.pytree import Base
from operator import mod
from unittest import result
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.base import clone, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    # This is equivalent to n_rows = ceil(len(instances) / images_per_row):
    n_rows = (len(instances) - 1) // images_per_row + 1

    # Append empty images to fill the end of the grid, if needed:
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)

    # Reshape the array so it's organized as a grid containing 28×28 images:
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))

    # Combine axes 0 and 2 (vertical image grid axis, and vertical image axis),
    # and axes 1 and 3 (horizontal axes). We first need to move the axes that we
    # want to combine next to each other, using transpose(), and only then we
    # can reshape:
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size,
                                                         images_per_row * size)
    # Now that we have a big image, we just need to show it:
    plt.imshow(big_image, cmap = mpl.cm.binary, **options)
    plt.axis("off")

#manual kfold cross validation
def manual_kfold(classifier, X_train, y_train):
    skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=16)
    results=[]
    for train_index, test_index in skfolds.split(X_train, y_train):
        clone_clf = clone(classifier)
        X_train_folds = X_train[train_index]
        y_train_folds = y_train[train_index]
        X_test_fold = X_train[test_index]
        y_test_fold = y_train[test_index]
        
        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred==y_test_fold)
        results.append(n_correct/len(y_pred))
    return results

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    plt.axis([-50000, 50000, 0, 1])             # Not shown

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)                                            # Not shown
 
#%% download MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()

# %%

X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)

some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()

print(y[0])

plt.figure(figsize=(9,9))
example_images = X[:100]
plot_digits(example_images, images_per_row=10)

plt.show()

#the label is a string, cast into integer
y = y.astype(np.uint8)
#mnist is already divided into train and test
X_train, X_test, y_train, y_test = X[:60000], X[60000:],y[:60000], y[60000:]
 
# %%  Stochastic Gradient Descendent
#lets develop a binary classifier, to recognize number 5
# lets transform the labels
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd = SGDClassifier(random_state=16)
sgd.fit(X_train, y_train_5)

sgd.predict([some_digit])
scores = cross_val_score(sgd, X_train, y_train_5, cv=3, scoring='accuracy')

#lets develop a dumb classifier to see the accuracy of this one
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5 = Never5Classifier()
scores2 = cross_val_score(never_5, X_train, y_train_5, cv=3, scoring='accuracy')
#accuracy is not a reliable metric
# %% confusion matrix

#cross_val_predict is almost the same of cross_val_score but it returns the vector with prediction
y_train_pred = cross_val_predict(sgd, X_train, y_train_5, cv=3)
conf_mat = confusion_matrix(y_train_5, y_train_pred)

y_train_perfect_predictions = y_train_5 #pretend to reach perfection
conf_mat2 = confusion_matrix(y_train_5, y_train_perfect_predictions)

#%% precision and recall
precision_score(y_train_5, y_train_pred)
#also like this
conf_mat[1, 1]/(conf_mat[0, 1] + conf_mat[1, 1])
recall_score(y_train_5, y_train_pred)
conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
f1_score(y_train_5, y_train_pred)
conf_mat[1, 1] / (conf_mat[1, 1] + (conf_mat[1, 0] + conf_mat[0, 1]) / 2)

#decision function return a score for each instance
y_scores = sgd.decision_function([some_digit])
treshold = 0
y_some_digit_pred = (y_scores > treshold)
y_some_digit_pred
#lets raise the treshold
threshold = 8000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

#how can we decide the treshold? 
y_scores = cross_val_predict(sgd, X_train, y_train_5, cv=3, method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
# %% 

#we aim for 90% precision or 90% recall
recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
y_train_pred_90 = (y_scores >= threshold_90_precision)
print(precision_score(y_train_5, y_train_pred_90))
print(recall_score(y_train_5, y_train_pred_90))

#with 90% precision, we have 59% recall
plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:")
plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
plt.plot([recall_90_precision], [0.9], "ro")

plt.show()

plt.figure(figsize=(8, 4))                                                                  
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")                 
plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")                                
plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")
plt.plot([threshold_90_precision], [0.9], "ro")                                             
plt.plot([threshold_90_precision], [recall_90_precision], "ro")                             
plt.show()
# %% ROC curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

plt.figure(figsize=(8, 6))                                    
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]           
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  
plt.plot([fpr_90], [recall_90_precision], "ro")               

plt.show()

print(roc_auc_score(y_train_5, y_scores))

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,method="predict_proba")

y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

recall_for_forest = tpr_forest[np.argmax(fpr_forest >= fpr_90)]

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")
plt.plot([fpr_90], [recall_90_precision], "ro")
plt.plot([fpr_90, fpr_90], [0., recall_for_forest], "r:")
plt.plot([fpr_90], [recall_for_forest], "ro")
plt.grid(True)
plt.legend(loc="lower right", fontsize=16)

plt.show()

print(roc_auc_score(y_train_5, y_scores_forest))
y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
print(precision_score(y_train_5, y_train_pred_forest))
print(recall_score(y_train_5, y_train_pred_forest))

# %% multiclass classification

#if you have a binary classifier there are two main approaches to multiclass classification:
# one-vs-one, one classifier every couple of labels ie 0 vs 1, 0 vs 2, 0 vs 3, ... in the end N(N-1)/2 classifiers
# one-vs-all, one classifier to distinguish one class from the others, ie one classifiers to classify 0 versus the other numbers, ...

#for SVM is better OvO, sklearn is already implemented like this

svm = SVC()
svm.fit(X_train, y_train)
#one prediction
print(svm.predict([some_digit]))
some_digit_scores = svm.decision_function([some_digit])
print(some_digit_scores)
print(np.argmax(some_digit_scores))
print(svm.classes_)
print(svm.classes_[5])

# %% OvA

ovr = OneVsRestClassifier(SVC())
ovr.fit(X_train, y_train)
ovr.predict([some_digit])
print(ovr.estimators_)
print(len(ovr.estimators_))

sgd.fit(X_train, y_train)
print(sgd.predict([some_digit]))
print(sgd.decision_function([some_digit]))

print(cross_val_score(sgd, X_train, y_train, cv=3, scoring='accuracy'))
#%%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
print(cross_val_score(sgd, X_train_scaled, y_train, cv=3, scoring='accuracy'))

# %% error analysis

#warning time
y_train_pred = cross_val_predict(sgd, X_train_scaled, y_train, cv=3)

#on the rows the actual class, columns the pred class
conf_mx = confusion_matrix(y_train, y_train_pred)
#%%
#to show the cm matrix in a readable way
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
#divide  each value in the cm by the number of images corresponding to the class
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

#fill the diagonal with zeros
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)

plt.show()

#%% multilabel

#lets divide the labels into odd and >7
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn = KNeighborsClassifier()
knn.fit(X_train, y_multilabel)
print(knn.predict([some_digit]))

y_train_knn_pred = cross_val_predict(knn, X_train, y_multilabel, cv=3)
print(f1_score(y_multilabel, y_train_knn_pred, average='macro'))

#%% multioutput

#lets build a system to remove noise from images
noise = np.random.randint(0, 100, (len(X_train)), 784)
X_train_mod = X_train + noise

noise = np.random.randint(0, 100, (len(X_test)), 784)
X_test_mod = X_test + noise

y_train_mod = X_train
y_test_mod = X_test

some_index = 0
plt.subplot(121); plot_digit(X_test_mod[some_index])
plt.subplot(122); plot_digit(y_test_mod[some_index])
plt.show()

#warning time
knn.fit(X_train_mod, y_train_mod)
clean_digit = knn.predict(X_test_mod[some_index])

#%% dummy classifier

from sklearn.dummy import DummyClassifier
dmy_clf = DummyClassifier(strategy="prior")
y_probas_dmy = cross_val_predict(dmy_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_dmy = y_probas_dmy[:, 1]
fprr, tprr, thresholdsr = roc_curve(y_train_5, y_scores_dmy)
plot_roc_curve(fprr, tprr)

#%% knn

#lets train a knn classifier
knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=4)
knn_clf.fit(X_train, y_train)
y_knn_pred = knn_clf.predict(X_test)
print(accuracy_score(y_test, y_knn_pred))

#shift for data aug
from scipy.ndimage.interpolation import shift
def shift_digit(digit_array, dx, dy, new=0):
    return shift(digit_array.reshape(28, 28), [dy, dx], cval=new).reshape(784)

plot_digit(shift_digit(some_digit, 5, 1, new=100))

X_train_expanded = [X_train]
y_train_expanded = [y_train]
for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    shifted_images = np.apply_along_axis(shift_digit, axis=1, arr=X_train, dx=dx, dy=dy)
    X_train_expanded.append(shifted_images)
    y_train_expanded.append(y_train)

X_train_expanded = np.concatenate(X_train_expanded)
y_train_expanded = np.concatenate(y_train_expanded)
X_train_expanded.shape, y_train_expanded.shape

knn_clf.fit(X_train_expanded, y_train_expanded)

y_knn_expanded_pred = knn_clf.predict(X_test)

accuracy_score(y_test, y_knn_expanded_pred)

ambiguous_digit = X_test[2589]
knn_clf.predict_proba([ambiguous_digit])

plot_digit(ambiguous_digit)

#%% mnist classifier with 97% accuracy
from sklearn.model_selection import GridSearchCV

param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3)
grid_search.fit(X_train, y_train)

grid_search.best_params_

grid_search.best_score_

from sklearn.metrics import accuracy_score

y_pred = grid_search.predict(X_test)
accuracy_score(y_test, y_pred)

