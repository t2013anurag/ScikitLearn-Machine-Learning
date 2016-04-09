# Using Decision tree
from sklearn import tree

# features = [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]
# labels = ["apple", "apple", "orange", "orange"]

# smooth : 1, bumpy : 0
# apple : 0, orange : 1

#features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# Checking with more features
# labels = [0, 0, 1, 1]
features = [[140, 1], [130, 1], [150, 0], [170, 0],[180, 1], [120, 1], [142, 0], [150, 0]]
labels = [0, 1, 1, 1, 1, 1, 1, 0]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print clf.predict([[145, 0]])