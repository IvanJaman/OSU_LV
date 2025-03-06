# Napišite program koji od korisnika zahtijeva unos radnih sati te koliko je placen ´
# po radnom satu. Koristite ugradenu Python metodu ¯ input(). Nakon toga izracunajte koliko ˇ
# je korisnik zaradio i ispišite na ekran. Na kraju prepravite rješenje na nacin da ukupni iznos ˇ
# izracunavate u zasebnoj funkciji naziva total_euro.

def paycheckCalculator(hours, rate):
    return hours * rate

print("Please input the amount of hours (e.g. 35):")
hours = float(input())
print("Please input the hourly rate (e.g. 8.5):")
rate = float(input())

paycheck = paycheckCalculator(hours, rate)
print(f"Your paycheck amounts to ${paycheck:.2f}")