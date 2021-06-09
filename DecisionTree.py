import sklearn
from sklearn import tree
features = [[1,120], [2,135], [3,145], [2,125], [2,130], [3,155], [1,100], [1,110], [1,105], [2,122], [3,160], [3,180]]
label = [1, 2, 2, 2, 2, 3, 1, 1, 1, 2, 3, 3]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,label)
print(clf.predict([[2,138]]))
