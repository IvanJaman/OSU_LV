''''
Napišite program koji ´ ce kreirati sliku koja sadrži ˇ cetiri kvadrata crne odnosno
 bijele boje (vidi primjer slike 2.4 ispod). Za kreiranje ove funkcije koristite numpy funkcije
 zeros i ones kako biste kreirali crna i bijela polja dimenzija 50x50 piksela. Kako biste ih složili
 u odgovaraju´ ci oblik koristite numpy funkcije hstack i vstack.
'''

import numpy as np
import matplotlib.pyplot as plt

whiteSquare = np.ones((50,50))
blackSquare = np.zeros((50,50))

row1 = np.hstack((blackSquare, whiteSquare))  
row2 = np.hstack((whiteSquare, blackSquare))

image = np.vstack((row1, row2))

plt.figure()
plt.imshow(image, cmap="gray", vmin=0, vmax=1)  
plt.show()