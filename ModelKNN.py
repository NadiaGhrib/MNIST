#Model KNN :

from DataKNN import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def Model():
    model = KNeighborsClassifier(n_neighbors=5)
    return model

model=Model()
model.fit(x_train, y_train)




