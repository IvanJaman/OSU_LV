import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3, 3, 1]
y = [1, 2, 2, 1, 1]

plt.plot(x, y, 'g', linewidth=1, marker=".", markersize=5)
plt.axis([0,4,0,4])
plt.xlabel('x')
plt.ylabel('vrijednosti funkcije')
plt.title('zad1')
plt.show()