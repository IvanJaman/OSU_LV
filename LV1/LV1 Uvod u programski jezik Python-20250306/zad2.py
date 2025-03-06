'''
Napišite program koji od korisnika zahtijeva upis jednog broja koji predstavlja
nekakvu ocjenu i nalazi se izmedu 0.0 i 1.0. Ispišite kojoj kategoriji pripada ocjena na temelju ¯
sljedecih uvjeta: 
    >= 0.9 A
    >= 0.8 B
    >= 0.7 C
    >= 0.6 D
    < 0.6 F
Ako korisnik nije utipkao broj, ispišite na ekran poruku o grešci (koristite try i except naredbe).
Takoder, ako je broj izvan intervala [0.0 i 1.0] potrebno je ispisati odgovaraju ¯ cu poruku
'''

print("Enter a grade from 0.0 to 1.0")

try:
    grade = float(input())  

    if grade < 0.0 or grade > 1.0:
        print("Error! Grade has to be in range 0.0 to 1.0!")
    else:
        if grade >= 0.9:
            category = 'A'
        elif grade >= 0.8:
            category = 'B'
        elif grade >= 0.7:
            category = 'C'
        elif grade >= 0.6:
            category = 'D'
        else:
            category = 'F'

        print(f"Your grade {grade} belongs to category {category}")

except ValueError:
    print("Error! You have to enter a valid grade (a number between 0.0 and 1.0).")

