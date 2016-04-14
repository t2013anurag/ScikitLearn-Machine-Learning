from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

print iris.feature_names 
#prints all the feautes like ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print iris.target_names #prints the target result ['setosa' 'versicolor' 'virginica']

#printing the data(features of first example)
print iris.data[0] #[ 5.1  3.5  1.4  0.2] 
# 5.1 is sepal length, 3.5 is sepal width, 1.4 is petal length and 0.2 is petal width

#printing the target of first example(result for the iris.data[0])
print iris.target[0]     # 0 for setosa, 1 for versicolor and 2 for virginica
