'''
The following code trains an SVM to classify the image files and create a confusion matrix expressing the results for
each classification class in the test set
'''

from sklearn import svm
import random
import math
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


# this function reads in the training examples and loads them into arrays
def read_training_data(X, y, filename, id):
    with open(filename, 'r') as f:
        for line in f:
            x = []
            for word in line.split():
                x.append(int(word))
            X.append(x)
            y.append(id)

# initialize the arrays for the training data
X = []
y = []

# read all the training data
read_training_data(X, y, 'pizza.txt', 1)
read_training_data(X, y, 'chicken.txt', 2)
read_training_data(X, y, 'burger.txt', 3)
read_training_data(X, y, 'burrito.txt', 4)

# shuffle the data to partition it into a training set and test set
X_shuf = []
y_shuf = []
index_shuf = range(len(X))
random.shuffle(index_shuf)
for i in index_shuf:
    X_shuf.append(X[i])
    y_shuf.append(y[i])

# train SVM
lin_clf = svm.LinearSVC()
train_length = int(math.floor(0.1*len(X_shuf)))
lin_clf.fit(X_shuf[:train_length], y_shuf[:train_length])

# print the accuracy of classifying the test set
print lin_clf.score(X_shuf[train_length:], y_shuf[train_length:])

# The following code is to create a confusion matrix visual
y_pred = lin_clf.predict(X_shuf[train_length:])
y_true = y_shuf[train_length:]
cm = confusion_matrix(y_true, y_pred)


# This code to plot the confusion marix is taken from
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
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
