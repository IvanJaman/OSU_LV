''''
 Pomo´ cu funkcija numpy.array i matplotlib.pyplot pokušajte nacrtati sliku
 2.3 u okviru skripte zadatak_1.py. Igrajte se sa slikom, promijenite boju linija, debljinu linije i
 sl
'''

import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3, 3, 1]
y = [1, 2, 2, 1, 1]

plt.plot(x, y, 'g', linewidth=2, marker=".", markersize=7)
plt.axis([0,4,0,4])
plt.xlabel('x')
plt.ylabel('vrijednosti funkcije')
plt.title('zad1')
plt.show()