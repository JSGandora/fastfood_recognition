from sklearn import svm
import random


def read_training_data(X, y, filename, id):
    with open(filename, 'r') as f:
        for line in f:
            x = []
            for word in line.split():
                x.append(int(word))
            X.append(x)
            y.append(id)


X = []
y = []

read_training_data(X, y, 'pizza.txt', 1)
read_training_data(X, y, 'chicken.txt', 2)
read_training_data(X, y, 'burger.txt', 3)
read_training_data(X, y, 'burrito.txt', 4)

X_shuf = []
y_shuf = []
index_shuf = range(len(X))
random.shuffle(index_shuf)
for i in index_shuf:
    X_shuf.append(X[i])
    y_shuf.append(y[i])

lin_clf = svm.LinearSVC()

import math
train_length = int(math.floor(0.1*len(X_shuf)))

lin_clf.fit(X_shuf[:train_length], y_shuf[:train_length])
print lin_clf.score(X_shuf[train_length:], y_shuf[train_length:])

# The following code is to create a confusion matrix visual

from sklearn.metrics import confusion_matrix

y_pred = lin_clf.predict(X_shuf[train_length:])
y_true = y_shuf[train_length:]
cm = confusion_matrix(y_true, y_pred)

import numpy as np
import matplotlib.pyplot as plt


# This code to plot the confusion marix is taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    target_names = ('Pizza', 'Chicken', 'Burger', 'Burrito')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()