'''
Napišite skriptu koja ´ce uˇcitati izgra ¯denu mrežu iz zadatka 1 i MNIST skup
podataka. Pomo´cu matplotlib biblioteke potrebno je prikazati nekoliko loše klasificiranih slika iz
skupa podataka za testiranje. Pri tome u naslov slike napišite stvarnu oznaku i oznaku predvi ¯denu
mrežom.
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test_s = x_test.astype("float32") / 255

x_test_s = np.expand_dims(x_test_s, -1)

model = load_model('mnist_model.keras') 

y_pred = model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1) 

incorrect_indices = np.where(y_pred_classes != y_test)[0]

num_images = 9
plt.figure(figsize=(10, 10))

for i in range(num_images):
    idx = incorrect_indices[i]  
    image = x_test[idx]  
    true_label = y_test[idx]  
    predicted_label = y_pred_classes[idx]
    
    plt.subplot(1, num_images, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'True: {true_label}\nPred: {predicted_label}')
    plt.axis('off')

plt.tight_layout()
plt.show()


