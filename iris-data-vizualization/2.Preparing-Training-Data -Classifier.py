import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
# print iris.feature_names 
# #prints all the feautes like ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# print iris.target_names #prints the target result ['setosa' 'versicolor' 'virginica']

# #printing the data(features of first example)
# print iris.data[0] #[ 5.1  3.5  1.4  0.2] 
# # 5.1 is sepal length, 3.5 is sepal width, 1.4 is petal length and 0.2 is petal width

# #printing the target of first example(result for the iris.data[0])
# print iris.target[0]     # 0 for setosa, 1 for versicolor and 2 for virginica

text_idx = [0, 50, 100] # that's the test data 0th row for 1st flower, 50th row for 2nd flower
# and 100th row for the third flower (0th feature row, 1st feature row and 2nd feature row)

#training data
train_target = np.delete(iris.target, text_idx)
train_data = np.delete(iris.data, text_idx, axis =0)

#testing data (it has the examples which we removed above)
test_target = iris.target[text_idx]
test_data = iris.data[text_idx]

#train the classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print test_target # gives [0, 1, 2] i.e we have one flower of each type
print clf.predict(test_data) # gives [0, 1, 2] which means the predicted values match the testing data
