from DataCNN import *
from ModelCNN import *


#Lecture des données


print("Train",train.shape)
print("Test",test.shape)


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


#Model CNN


#Evaluation

loss, acc = Model.evaluate(x_train, y_train, verbose=0)
print("loss: {0:.4f},  accuracy: {1:.4f}".format(loss, acc))


#Loss et Acc des train et test

plt.figure()
plt.subplot(2,1,1)
plt.plot(Model.history['accuracy'])
plt.plot(Model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.subplot(2,1,2)
plt.plot(Model.history['loss'])
plt.plot(Model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
figure