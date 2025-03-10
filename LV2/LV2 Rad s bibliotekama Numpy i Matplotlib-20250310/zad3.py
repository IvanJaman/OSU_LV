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

img = plt.imread("road.jpg")
img = img[:,:,0].copy()

plt.figure()
plt.imshow(img, cmap="gray")
plt.show()

# a) dio
brightImage = img + 50
brightImage[brightImage > 255] = 255
plt.figure()
plt.imshow(brightImage, cmap="gray")
plt.show()

# b) dio