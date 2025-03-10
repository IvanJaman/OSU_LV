''''
Skripta zadatak_3.py uˇ citava sliku ’road.jpg’. Manipulacijom odgovaraju´ ce
 numpy matrice pokušajte:
 a) posvijetliti sliku,
 b) prikazati samo drugu ˇ cetvrtinu slike po širini,
 c) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu,
 d) zrcaliti sliku.
'''

import numpy as np
import matplotlib.pyplot as plt

image = plt.imread("road.jpg")

def start():
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.show()

# a) dio
def brighten():
    brightImage = image + 50
    brightImage[brightImage > 255] = 255
    plt.figure()
    plt.imshow(brightImage, cmap="gray")
    plt.show()

# b) dio
def crop():
    x = image.shape[0]
    y = image.shape[1]

    startX = x // 2
    startY = y // 2
    croppedImage = image[startX:, startY:]

    plt.figure()
    plt.imshow(croppedImage, cmap="gray")
    plt.show()

# c) dio
def rotate():
    rotatedImage = np.rot90(image, k=1, axes=(1, 0))

    plt.figure()
    plt.imshow(rotatedImage, cmap="gray")
    plt.show()

# d) dio
def mirror():
    mirrorImage = np.fliplr(image)

    plt.figure()
    plt.imshow(mirrorImage, cmap="gray")
    plt.show()

