'''
Napišite skriptu koja ´ce uˇcitati izgra ¯denu mrežu iz zadatka 1. Nadalje, skripta
treba uˇcitati sliku test.png sa diska. Dodajte u skriptu kod koji ´ce prilagoditi sliku za mrežu,
klasificirati sliku pomo´cu izgra ¯dene mreže te ispisati rezultat u terminal. Promijenite sliku
pomo´cu nekog grafiˇckog alata (npr. pomo´cu Windows Paint-a nacrtajte broj 2) i ponovo pokrenite
skriptu. Komentirajte dobivene rezultate za razliˇcite napisane znamenke.
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image

model = load_model('mnist_model.keras') 

img_path = 'test.png'  
img = Image.open(img_path).convert('L')  

plt.imshow(img, cmap='gray')
plt.title("Originalna slika")
plt.axis('off')
plt.show()

img = img.resize((28, 28))

img_array = np.array(img)

img_array = img_array.astype("float32") / 255

img_array = np.expand_dims(img_array, axis=-1) 

img_array = np.expand_dims(img_array, axis=0) 

predictions = model.predict(img_array)

predicted_class = np.argmax(predictions, axis=1)  

print(f"Predviđena oznaka: {predicted_class[0]}")

for i in range(10):
    print(f"Klasa {i}: {predictions[0][i]:.4f}")