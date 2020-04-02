
#Lecture des données
from DataCNN import *

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense,Flatten,
from keras.layers import Conv2D

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


#Model CNN

model = Sequential()
model.add (Conv2D (64, kernel_size = (5,5), activation = "relu", input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add (Conv2D (32, kernel_size = (5,5), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add (Flatten ())
model.add (Dense (10, activation = "softmax"))


model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
Model = model.fit (x_train, y_train, validation_data = (x_test, y_test), epochs = 10)




