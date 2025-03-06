'''
Napišite Python skriptu koja ´ ce uˇ citati tekstualnu datoteku naziva SMSSpamCollection.txt
 [1]. Ova datoteka sadrži 5574 SMS poruka pri ˇ cemu su neke oznaˇ cene kao spam, a neke kao ham.
 Primjer dijela datoteke:
    ham Yupnextstop.
    ham Oklar... Joking wif u oni...
    spam Didyouhear about the new "Divorce Barbie"? It comes with all of Ken’s stuff!
a) Izraˇcunajte koliki je prosjeˇcan broj rijeˇci u SMS porukama koje su tipa ham, a koliko je
 prosjeˇ can broj rijeˇ ci u porukama koje su tipa spam.
 b) Koliko SMS poruka koje su tipa spam završava uskliˇ cnikom ?
'''

hamCounter = 0
spamCounter = 0
exclamationCounter = 0
lineCounter = 0

with open('LV1\LV1 Uvod u programski jezik Python-20250306\SMSSpamCollection.txt', 'r', encoding='utf-8') as spamMessages:
    for line in spamMessages:
        lineCounter += 1

        if(line.startswith('ham')):
            hamCounter += 1

        else:
            spamCounter += 1
            if(line.endswith('!')):
                exclamationCounter += 1
                
averageHam = (hamCounter / lineCounter) * 100
averageSpam = (spamCounter / lineCounter) * 100

print(f"There are, on average, {averageHam:.2f}% HAM messages in SMS file.")
print(f"There are, on average, {averageSpam:.2f}% SPAM messages in SMS file.")
print(f"There are {exclamationCounter} SPAM messages that end with '!' mark.")