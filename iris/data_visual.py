# %matplotlib inline
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
plt.style.use('seaborn')

cols = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
iris_data = pd.read_csv('data/iris.data', names=cols)

colors = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864']

# box and whisker plots
iris_data.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

# violin plot
f1, axes = plt.subplots(2, 2, figsize=(8, 8))
sns.despine(left=True)

sns.violinplot(x='species', y='sepal-length', data=iris_data, palette=colors, ax=axes[0, 0])
sns.violinplot(x='species', y='sepal-width', data=iris_data, palette=colors, ax=axes[0, 1])
sns.violinplot(x='species', y='petal-length', data=iris_data, palette=colors, ax=axes[1, 0])
sns.violinplot(x='species', y='petal-width', data=iris_data, palette=colors, ax=axes[1, 1])
plt.show()

# point plot
f2, axes = plt.subplots(2, 2, figsize=(8, 8))
sns.despine(left=True)

sns.pointplot(x='species', y='sepal-length', data=iris_data, palette=colors, ax=axes[0, 0])
sns.pointplot(x='species', y='sepal-width', data=iris_data, palette=colors, ax=axes[0, 1])
sns.pointplot(x='species', y='petal-length', data=iris_data, palette=colors, ax=axes[1, 0])
sns.pointplot(x='species', y='petal-width', data=iris_data, palette=colors, ax=axes[1, 1])
plt.show()

# pair plot
sns.pairplot(data=iris_data, palette=colors, hue='species')
plt.show()
