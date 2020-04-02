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


print(np.unique(y_train))
#Le jeu de donnée comporte 9 classes










