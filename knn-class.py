import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics

#X = datos[['RU']]
#Y= datos[['ID']]
#neigh = KNeighborsRegressor(n_neighbors=3)
#neigh.fit(X, Y)
#
#datonuevo = pd.DataFrame([{'RU':150}])
#neigh.predict(datonuevo)

iris = load_iris()
iris_df = sns.load_dataset("iris")

x = iris.data
y = iris.target

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x,y)
Y_pred = model.predict(x)
accuracy_score=metrics.accuracy_score(y,Y_pred)
confusion_matrix=metrics.confusion_matrix(y,Y_pred)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,stratify=y)


model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
Y_pred = model.predict(x_test)

accuracy_score_split = metrics.accuracy_score(y_test,Y_pred)
confusion_matrix_split = metrics.confusion_matrix(y_test,Y_pred)
