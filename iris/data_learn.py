import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


def train_and_predict(dataset, data_cols):
    # samples and features
    X = dataset[data_cols]
    y = dataset['species']

    # encode features
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # split training data and testing data by the ratio 75:25
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=101)
    print("train: ", train_X.shape, train_y.shape, "\ntest: ", test_X.shape, test_y.shape)

    # Support Vector Machine
    model = svm.SVC()
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    print('The accuracy of the SVM is: {0}'.format(metrics.accuracy_score(prediction, test_y)))

    # Logistic Regression
    model = LogisticRegression()
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    print('The accuracy of the Logistic Regression is: {0}'.format(metrics.accuracy_score(prediction, test_y)))

    # Decision Tree
    model = DecisionTreeClassifier()
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    print('The accuracy of the Decision Tree is: {0}'.format(metrics.accuracy_score(prediction, test_y)))

    # K-Nearest Neighbours
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    print('The accuracy of the KNN is: {0}'.format(metrics.accuracy_score(prediction, test_y)))


cols = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
iris_data = pd.read_csv('data/iris.data', names=cols)
train_and_predict(iris_data, ['sepal-length', 'sepal-width', 'petal-length', 'petal-width'])
# train_and_predict(iris_data, ['sepal-width'])
