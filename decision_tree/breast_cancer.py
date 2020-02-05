from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


def load_data():
    cancer = load_breast_cancer()
    return train_test_split(cancer.data, cancer.target, stratify=cancer.target, test_size=0.25, random_state=101)


def decision_tree_predict(decision_tree_classifier, train_test_data):
    train_X, test_X, train_y, test_y = train_test_data
    decision_tree_classifier.fit(train_X, train_y)
    prediction = decision_tree_classifier.predict(test_X)
    return prediction, metrics.accuracy_score(prediction, test_y)


cancer_data = load_data()
print("Actual values: \n{}".format(cancer_data[3]))

# default settings
default_tree = DecisionTreeClassifier()
predict, score = decision_tree_predict(default_tree, cancer_data)
print('Prediction: \n{}'.format(predict))
print('The accuracy with default settings is: {0}'.format(score))

# pre-pruning settings
pre_pruning_tree = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=3, min_samples_leaf=5)
predict, score = decision_tree_predict(pre_pruning_tree, cancer_data)
print('Prediction: \n{}'.format(predict))
print('The accuracy with pre-pruning settings is: {0}'.format(score))
