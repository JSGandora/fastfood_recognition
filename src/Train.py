from sklearn import svm
import random


def read_training_data(X, y, filename, id):
    with open(filename,'r') as f:
        for line in f:
            x = []
            for word in line.split():
                x.append(int(word))
            X.append(x)
            y.append(id)

X=[]
y=[]

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
lin_clf.fit(X_shuf[:len(X_shuf)/2], y_shuf[:len(X_shuf)/2])