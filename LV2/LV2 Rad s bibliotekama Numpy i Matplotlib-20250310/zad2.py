''''
Datoteka data.csv sadrži mjerenja visine i mase provedena na muškarcima i
 ženama. Skripta zadatak_2.py uˇ citava dane podatke u obliku numpy polja data pri ˇ cemu je u
 prvom stupcu polja oznaka spola (1 muško, 0 žensko), drugi stupac polja je visina u cm, a tre´ ci
 stupac polja je masa u kg

a) Na temelju veliˇ cine numpy polja data, na koliko osoba su izvršena mjerenja?
 b) Prikažite odnos visine i mase osobe pomo´ cu naredbe matplotlib.pyplot.scatter.
 c) Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici.
 d) Izraˇ cunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom
 podatkovnom skupu.
 e) Ponovite zadatak pod d), ali samo za muškarce, odnosno žene. Npr. kako biste izdvojili
 muškarce, stvorite polje koje zadrži bool vrijednosti i njega koristite kao indeks retka.
 ind = (data[:,0] == 1)
'''

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.csv', delimiter=',', skiprows=1)

count = data.shape[0]
print(f"a) Mjerenje je izvršeno na {count} osoba")

# b) dio
height = data[:, 1]  
weight = data[:, 2]

plt.scatter(height, weight, color='b', marker='.', s=2) 
plt.axis([130,210,20,150])
plt.xlabel('Visina [cm]')
plt.ylabel('Težina [kg]')
plt.title('zad2-b')
plt.show()

# c) dio
height50th = height[::50]
weight50th = weight[::50]

plt.scatter(height50th, weight50th, color='g', marker='.', s=10) 
plt.axis([130,210,20,150])
plt.xlabel('Visina [cm]')
plt.ylabel('Težina [kg]')
plt.title('zad2-c')
plt.show()

# d) dio
minHeight = np.min(height)
maxHeight = np.max(height)
averageHeight = np.mean(height)

print(f"Min. visina: {minHeight} cm; Max. visina: {maxHeight} cm; Prosječna visina: {averageHeight:.2f} cm")

# e) dio
ind = (data[:,0] == 1)
maleHeight = data[ind, 1]

minMaleHeight = np.min(maleHeight)
maxMaleHeight = np.max(maleHeight)
averageMaleHeight = np.mean(maleHeight)

print(f"Min. visina muškaraca: {minMaleHeight} cm; Max. visina muškaraca: {maxMaleHeight} cm; Prosječna visina muškaraca: {averageMaleHeight:.2f} cm")
