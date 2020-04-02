#Lecture des données

from DataKNN import *
from ModelKNN import *

#Prédiction

def Y_pred():
    y_pred = model.predict(x_test[:100])
    return y_pred

y_pred=Y_pred()

#Evaluation

def ConMatrix():
    m = confusion_matrix(y_true=y_test[:100], y_pred=y_pred)
    return m

m=ConMatrix()


tp = np.diag(m)
def Acc():
    acc= np.sum(tp) / 100 * 100
    return acc

acc = Acc()





