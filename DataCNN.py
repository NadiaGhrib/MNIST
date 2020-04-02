import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Lecture des données

def Train() :
    train = pd.read_csv("/Users/nadiaghrib/Desktop/ML/train.csv")
    return train

def Test() :
    test = pd.read_csv("/Users/nadiaghrib/Desktop/ML/test.csv")
    return test

train= Train()
test = Test()

#Préparation des données
    #Train

def Y_train():
    y_train = train["label"]
    return y_train

def X_train():
    x_train = train.drop(labels = ["label"],axis = 1)
    return x_train

y_train=Y_train()
x_train =X_train()

#On split le data set : test size 10% et train size 90%

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.1, random_state=2)


#Normalise les données

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#Reshape les données

x_train = x_train.values.reshape((-1, 28, 28, 1))
x_test = x_test.values.reshape((-1, 28, 28, 1))

# One hot labels.
# Cela signifie qu'une colonne sera créée pour chaque catégorie de sortie et une variable binaire est entrée pour chaque catégorie.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Les deux premières lignes :
#premier nombre est le nombre d'images (37 800 pour x_train et 42 000 pour x_test).
#Deuxième et troisième nombre la forme de chaque image (28x28).
#Le dernier nombre est 1, ce qui signifie que les images sont en niveaux de gris.





