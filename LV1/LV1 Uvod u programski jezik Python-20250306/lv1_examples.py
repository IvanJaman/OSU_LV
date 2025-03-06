def pr1():    
    x = 23
    print ( x )
    x = x + 7
    print ( x ) # komentar : ispis varijable na ekranu

def pr2():
    lstEmpty = [ ]
    lstFriend = ['Marko', 'Luka', 'Pero']
    lstFriend . append ('Ivan')
    print ( lstFriend [0])
    print ( lstFriend [0:1:2])
    print ( lstFriend [ :2])
    print ( lstFriend [1: ])
    print ( lstFriend [1:3])

def pr3():
    i = 5
    while i > 0:
        print ( i )
        i = i - 1
    print (" Petlja gotova ")
    for i in range (0 , 5 ):
        print ( i )

def pr4():
    a = [1 , 2 , 3]
    b = [4 , 5 , 6]
    c = a + b
    print ( c )
    print ( max ( c) )
    c[0] = 7
    c . pop ()
    for number in c:
        print ('List number', number )
    print ('Done !')

def pr5():
    fruit = 'banana'
    index = 0
    count = 0
    while index < len ( fruit ):
        letter = fruit [ index ]
        if letter == 'a':
            count = count + 1

        print ( letter )
        index = index + 1

    print ( count )

    print ( fruit [0:3])
    print ( fruit [0: ])
    print ( fruit [2:6:1])
    print ( fruit [0: -1])

def pr6():
    line = 'Dobrodosli u nas grad'

    if( line.startswith ('Dobrodosli') ):
        print ('Prva rijec je Dobrodosli')
    elif ( line.startswith ('dobrodosli') ):
        print ('Prva rijec je dobrodosli')
               
    line2 = line.lower ()
    print ( line2 )

    data = 'From : pero@yahoo . com'
    atpos = data.find('@')
    print (atpos)

def pr7():
    letters = ('a', 'b', 'c', 'd', 'e')
    numbers = (1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 )
    mixed = (1 , 'Hello', 3.14)

    print ( letters [0])
    print ( letters [1:4])
    for letter in letters :
        print ( letter )

def pr8():
    hr_num = {'jedan':1 , 'dva':2 , 'tri':3}
              
    print ( hr_num )
    print ( hr_num ['dva'])
                    
    hr_num ['cetiri'] = 4
    print ( hr_num )

def pr9():
    fhand = open ('LV1 Uvod u programski jezik Python-20250306\example.txt')
    for line in fhand:
        line = line.rstrip ()
        print ( line )
        words = line.split ()
    fhand . close ()

pr9()
