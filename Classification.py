import pandas as pd
df = pd.read_csv('Iris.csv')
print(df)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#print(df.isnull().sum())
print("Shape is:")
print(df.shape)
print("Short Description:")
print(df.describe())

y = df['Species'].values
print(y)
df.drop('Species', axis=1, inplace=True)
x = df.values
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
clf = KNeighborsClassifier()
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)
print(accuracy_score(y_test, predictions))
