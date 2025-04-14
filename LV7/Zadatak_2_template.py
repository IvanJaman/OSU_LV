'''
Kvantizacija boje je proces smanjivanja broja razliˇcitih boja u digitalnoj slici, ali
uzimaju´ci u obzir da rezultantna slika vizualno bude što sliˇcnija originalnoj slici. Jednostavan
naˇcin kvantizacije boje može se posti´ci primjenom algoritma K srednjih vrijednosti na RGB
vrijednosti elemenata originalne slike. Kvantizacija se tada postiže zamjenom vrijednosti svakog
elementa originalne slike s njemu najbližim centrom. Na slici 7.3a dan je primjer originalne
slike koja sadrži ukupno 106,276 boja, dok je na slici 7.3b prikazana rezultantna slika nakon
kvantizacije i koja sadrži samo 5 boja koje su odre ¯dene algoritmom K srednjih vrijednosti.
1. Otvorite skriptu zadatak_2.py . Ova skripta uˇcitava originalnu RGB sliku test_1.jpg
te ju transformira u podatkovni skup koji dimenzijama odgovara izrazu (7.2) pri ˇcemu je n
broj elemenata slike, a m je jednak 3. Koliko je razliˇcitih boja prisutno u ovoj slici?
2. Primijenite algoritam K srednjih vrijednosti koji ´ce prona´ci grupe u RGB vrijednostima
elemenata originalne slike.
3. Vrijednost svakog elementa slike originalne slike zamijeni s njemu pripadaju´cim centrom.
4. Usporedite dobivenu sliku s originalnom. Mijenjate broj grupa K . Komentirajte dobivene
rezultate.
5. Primijenite postupak i na ostale dostupne slike.
6. Grafiˇcki prikažite ovisnost J o broju grupa K . Koristite atribut inertia objekta klase
KMeans. Možete li uoˇciti lakat koji upu´cuje na optimalni broj grupa?
7. Elemente slike koji pripadaju jednoj grupi prikažite kao zasebnu binarnu sliku. Što
primje´cujete?
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.image as mpimg

img = mpimg.imread("LV7/imgs/test_6.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
if img.max() > 1.0:
    img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

# a) dio
unique_colors = np.unique(img_array, axis=0)
print(f'Broj različitih RGB boja na slici: {len(unique_colors)}')

# b) dio
k = 5
km = KMeans(n_clusters=k, init='random', n_init=5, random_state=0)
labels = km.fit_predict(img_array)

# c) dio
centers = km.cluster_centers_ 
labels = km.labels_ 
img_array_aprox = centers[labels].reshape(w, h, d)

plt.figure()
plt.title(f"Slika s {k} dominantnih boja")
plt.imshow(img_array_aprox)
plt.tight_layout()
plt.show()

# d) dio
print("Dobivena slika ima samo 2 boje za k=2, no i dalje jasno prikazuje broj registarskih oznaka, prema tome čak je i to dovoljno.")
print("Slika s 5 unikatnih boja vrlo je slična orginalnoj slici.")

# e) dio
print("Primjenio.")

# f) dio
inertias = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(img_array)
    inertias.append(kmeans.inertia_)  

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertias, marker='o', color='b', linestyle='-', markersize=8)
plt.title('Ovisnost inertia-e o broju klastera (K)')
plt.xlabel('Broj klastera (K)')
plt.ylabel('Inertia')
plt.xticks(range(1, 11))  
plt.grid(True)
plt.tight_layout()
plt.show()

# g) dio
for i in range(1, k):
    group_label = i
    binary_image = (labels == group_label).reshape(w, h)

    plt.figure(figsize=(8, 6))
    plt.title(f'Binarna slika za grupu {group_label}')
    plt.imshow(binary_image, cmap='gray')
    plt.tight_layout()
    plt.show()