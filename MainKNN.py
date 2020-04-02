from DataKNN import *
from ModelKNN import *
from EvaluationKNN import *


#Lecture des données

print("Train",train.shape)
print("Test",test.shape)


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


#Model KNN

print("Classes du modèle", model.classes_)
#Permet de vérifier que les classes récupérées par le modèle sont cohérentes avec la vérité terrain.


#Evaluation

tp = np.diag(m)

print("Accuracy : ",np.sum(tp)/100*100)





