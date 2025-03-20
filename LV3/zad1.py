'''
Skripta zadatak_1.py uˇcitava podatkovni skup iz data_C02_emission.csv.
 Dodajte programski kod u skriptu pomo´ cu kojeg možete odgovoriti na sljede´ ca pitanja:
 a) Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka veliˇcina? Postoje li izostale ili
 duplicirane vrijednosti? Obrišite ih ako postoje. Kategoriˇcke veliˇcine konvertirajte u tip
 category.
 b) Koja tri automobila ima najve´ cu odnosno najmanju gradsku potrošnju? Ispišite u terminal:
 ime proizvo¯ daˇ ca, model vozila i kolika je gradska potrošnja.
 c) Koliko vozila ima veliˇcinu motora izme¯ du 2.5 i 3.5 L? Kolika je prosjeˇcna C02 emisija
 plinova za ova vozila?
 d) Koliko mjerenja se odnosi na vozila proizvo¯ daˇca Audi? Kolika je prosjeˇcna emisija C02
 plinova automobila proizvo¯ daˇ ca Audi koji imaju 4 cilindara?
 e) Koliko je vozila s 4,6,8... cilindara? Kolika je prosjeˇ cna emisija C02 plinova s obzirom na
 broj cilindara?
 f) Kolika je prosjeˇ cna gradska potrošnja u sluˇ caju vozila koja koriste dizel, a kolika za vozila
 koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
 g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najve´ cu gradsku potrošnju goriva?
 h) Koliko ima vozila ima ruˇ cni tip mjenjaˇ ca (bez obzira na broj brzina)?
 i) Izraˇ cunajte korelaciju izme¯ du numeriˇ ckih veliˇ cina. Komentirajte dobiveni rezultat
'''

import pandas as pd
import numpy as np

data = pd.read_csv('data_C02_emission.csv')

def a(data):
    print(data)

    print(f"Dataset sadrži {len(data)} mjerenja.")

    print(f"Tipovi podataka: ")
    print(data.info())

    print(data.isnull())
    data.dropna()
    data.drop_duplicates()
    data = data.reset_index(drop=True)
    print(f"Dataset sada sadrži {len(data)} mjerenja.")

    grouped_by_vehicle_class = data.groupby('Vehicle Class')
    print(grouped_by_vehicle_class.size())
    grouped_by_cylinders = data.groupby('Cylinders')
    print(grouped_by_cylinders.size())
    grouped_by_transmission = data.groupby('Transmission')
    print(grouped_by_transmission.size())
    grouped_by_fuel_type = data.groupby('Fuel Type')
    print(grouped_by_fuel_type.size())
    
def b(data):
    print()
    print(f"Najveća gradska potrošnja: ")
    largest = data.sort_values("Fuel Consumption City (L/100km)", ascending=False)
    print(largest[['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(3))
    
    print()
    print(f"Najmanja gradska potrošnja: ")
    smallest = data.sort_values("Fuel Consumption City (L/100km)", ascending=True)
    print(smallest[['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(3))
    
def c(data):
    count = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
    print(f"{len(count)} vozila ima veličinu motora između 2.5 i 3.5 L")

    co2 = count['CO2 Emissions (g/km)'].mean()
    print(f"Prosječna CO2 emisija je {co2:.2f} g/km")
    
def d(data):
    audi = data[(data['Make'] == 'Audi')]
    print(f"{len(audi)} mjerenja odnosi se na proizvođača Audi")
    
    audi_4_cylinders = data[(data['Make'] == 'Audi') & (data['Cylinders'] == 4)]
    audi_4_cylinders_co2_emissions = audi_4_cylinders['CO2 Emissions (g/km)'].mean()
    print(f"Prosječna CO2 emisija Audija s 4 cilindra je {audi_4_cylinders_co2_emissions:.2f} g/km")

def e(data):
    four_or_more_cylinders = data[(data['Cylinders'] >= 4) & (data['Cylinders'] % 2 == 0)]
    print(f"{len(four_or_more_cylinders)} automobila ima 4, 6, 8... cilindara")
    
    grouped_by_cylinders = four_or_more_cylinders.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
    
    print(grouped_by_cylinders)

def f(data):
    diesel_fueled = data[(data['Fuel Type'] == 'D')]
    regular_petrol_fueled = data[(data['Fuel Type'] == 'X')]

    print(f"Prosječna gradska potrošnja dizel automobila je {diesel_fueled['Fuel Consumption City (L/100km)'].mean():.2f}")
    print(f"Prosječna gradska potrošnja regularni benzin automobila je {regular_petrol_fueled['Fuel Consumption City (L/100km)'].mean():.2f}")

def g(data):
    four_cylinders_diesel_vehicles = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')]
    largest_fuel_consumption_vehicle = four_cylinders_diesel_vehicles.loc[(four_cylinders_diesel_vehicles['Fuel Consumption City (L/100km)'].idxmax())]
    print(f"Automobil s 4 cilindra koji koristi dizel i najviše troši u gradu je {largest_fuel_consumption_vehicle['Make']} {largest_fuel_consumption_vehicle['Model']}, {largest_fuel_consumption_vehicle['Fuel Consumption City (L/100km)']} L/100km.")

def h(data):
    manuals = data[(data['Transmission'] == 'M')]
    print(f"Automobila s manualnim prijenosom ima {len(manuals)}")

def i(data):
    numeric_values = data[['Engine Size (L)', 'Cylinders', 'Fuel Consumption Comb (L/100km)', 'CO2 Emissions (g/km)']]

    correlation_matrix = numeric_values.corr()
    print(correlation_matrix)

    print("Vidimo veliku ovisnost između veličine motora u L, broja cilindara, potrošnje goriva i CO2 emisija (propocionalne su)")

i(data)