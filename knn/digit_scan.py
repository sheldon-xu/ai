import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


N = 32


def file2vector(filename):
    vector = np.ndarray((1, N*N), dtype=int)
    file = open(filename)
    for i in range(N):
        line = file.readline()
        for j in range(N):
            index = N*i+j
            vector[0, index] = int(line[j])
    return vector


def load_numbers_data(data_path):
    numbers_ds = np.ndarray((0, N*N), dtype=int)
    number_labels = []
    for filename in os.listdir(data_path):
        vector = file2vector(data_path + filename)
        numbers_ds = np.concatenate([numbers_ds, vector])
        number_labels.append(filename[0])
    return numbers_ds, number_labels


train_X, train_y = load_numbers_data("data/trainingDigits/")
test_X, test_y = load_numbers_data("data/testDigits/")

kNN = KNeighborsClassifier(n_neighbors=3)
kNN.fit(train_X, train_y)
prediction = kNN.predict(test_X)
print('The accuracy of the KNN is: {0}'.format(metrics.accuracy_score(prediction, test_y)))
