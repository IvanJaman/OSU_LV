import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Flatten, Dense
from tensorflow.keras.datasets import mnist


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f'Labela: {y_train[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

image = x_train[0]  
label = y_train[0]  

plt.imshow(image, cmap='gray')  
plt.title(f'Oznaka: {label}')  
plt.axis('off')  
plt.show()

print(f'Oznaka ove slike je: {label}')

# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = Sequential()

model.add(Flatten(input_shape=(28 * 28, 1)))

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.summary()


# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TODO: provedi ucenje mreze
epochs = 10
batch_size = 64
history = model.fit(x_train_s, y_train_s, epochs=epochs, batch_size=batch_size, validation_data=(x_test_s, y_test_s))

# TODO: Prikazi test accuracy i matricu zabune
test_loss, test_acc = model.evaluate(x_test_s, y_test_s, verbose=2)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

y_pred = model.predict(x_test_s)  
y_pred_classes = np.argmax(y_pred, axis=1)  
y_true = np.argmax(y_test_s, axis=1)  

cm = confusion_matrix(y_true, y_pred_classes)

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(10, 7))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.show()

# TODO: spremi model
model.save('mnist_model.keras')
