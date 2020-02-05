import pandas as pd

cols = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
iris_data = pd.read_csv('data/iris.data', names=cols)

print("head ---------------------------------: \n", iris_data.head())
print("describe -----------------------------: \n", iris_data.describe())
print("iris_data shape ----------------------: \n", iris_data.shape)
print("group by class -----------------------: \n", iris_data.groupby('species').size())
