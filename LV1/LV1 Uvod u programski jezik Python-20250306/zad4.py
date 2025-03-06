'''
Napišite Python skriptu koja ´ce uˇcitati tekstualnu datoteku naziva song.txt.
Potrebno je napraviti rjeˇcnik koji kao kljuˇceve koristi sve razliˇcite rijeˇci koje se pojavljuju u
datoteci, dok su vrijednosti jednake broju puta koliko se svaka rijeˇ c (kljuˇ c) pojavljuje u datoteci.
Koliko je riječi koje se pojavljuju samo jednom u datoteci? Ispišite ih.
'''

import string

word_count = {}
with open('LV1/LV1 Uvod u programski jezik Python-20250306/song.txt', 'r', encoding='utf-8') as song:
    for line in song:
        line = line.lower().translate(str.maketrans('', '', string.punctuation))
        words = line.split()
        for word in words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

print('Rječnik riječi i njihovih učestalosti:')
print(word_count)

unique_words = []
for word in word_count:
    if word_count[word] == 1: 
        unique_words.append(word)
        
print(f'Broj riječi koje se pojavljuju samo jednom: {len(unique_words)}')
print('Te riječi su:', unique_words)
