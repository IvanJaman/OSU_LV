'''
Napišite program koji od korisnika zahtijeva unos brojeva u beskonacnoj petlji 
sve dok korisnik ne upiše „Done“ (bez navodnika). Pri tome brojeve spremajte u listu. Nakon toga
potrebno je ispisati koliko brojeva je korisnik unio, njihovu srednju, minimalnu i maksimalnu
vrijednost. Sortirajte listu i ispišite je na ekran. Dodatno: osigurajte program od pogrešnog unosa
(npr. slovo umjesto brojke) na nacin da program zanemari taj unos i ispiše odgovaraju ˇ cu poruku.
'''

import statistics

numbers = []

print("Start entering numbers. When done, input 'done':")

while True:
    number = input()

    if(number.lower() == 'done'):
        if len(numbers) > 0:
            print(f"The user inputed {len(numbers)} numbers")
            print(f"The largest number was {max(numbers)}")
            print(f"The smallest number was {min(numbers)}")
            print(f"The average of all numbers was {statistics.mean(numbers)}")

            numbers.sort() 
            print(f"Sorted numbers: {numbers}")
        else:
            print("No numbers were entered.")
        break

    elif number.isalpha():  
        print("Error! You have to input numbers!")

    else:
        try:
            number = float(number)
            numbers.append(number)
        except ValueError:
            print("Error! You have to input a valid number!")