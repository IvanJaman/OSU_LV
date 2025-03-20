''''
 Napišite programski kod koji ´ ce prikazati sljede´ ce vizualizacije:
 a) Pomo´ cu histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz.
 b) Pomo´cu dijagrama raspršenja prikažite odnos izme¯ du gradske potrošnje goriva i emisije
 C02 plinova. Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose izme¯ du
 veliˇ cina, obojite toˇ ckice na dijagramu raspršenja s obzirom na tip goriva.
 c) Pomo´ cu kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip
 goriva. Primje´ cujete li grubu mjernu pogrešku u podacima?
 d) Pomo´cu stupˇcastog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu
 groupby.
 e) Pomo´ cu stupˇ castog grafa prikažite na istoj slici prosjeˇ cnu C02 emisiju vozila s obzirom na
 broj cilindara
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data_C02_emission.csv')

def a(data):
    plt.hist(data['Fuel Consumption City (L/100km)'], bins=13, color='blue', edgecolor='black') # broj binsa izračunat korištenjem sturgesove formule

    plt.xlabel('Fuel Consumption City (L/100km)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Fuel Consumption in City')

    plt.show()

def b(data):
    plt.scatter(data['Fuel Consumption City (L/100km)'], data['CO2 Emissions (g/km)'], color = 'blue', marker = 'o') 

    plt.xlabel('Fuel Consumption City (L/100km)')
    plt.ylabel('CO2 Emissions (g/km)')
    plt.title('Dependency of fuel consumption and CO2 emmisions')

    plt.show()

    print("Ovisnost CO2 emisija o potrošnji goriva je linearna, što znači da raste s povećanjem potrošnje goriva.")

def c(data):
    sns.boxplot(x='Fuel Type', y='Fuel Consumption Hwy (L/100km)', data=data)

    plt.xlabel('Fuel Type')
    plt.ylabel('Fuel Consumption Hwy (L/100km)')
    plt.title('Fuel Consumption by Fuel Type')

    plt.show()

    print("Vozia koja troše etanol troše u prosjeku najviše. Dizel ima manji raspon vrijednosti od ostalih, dok regularni benzin ima najveću. Premium benzin ima najviše ekstrema.")

def d(data):
    import matplotlib.pyplot as plt

    grouped_by_fuel_type = data.groupby('Fuel Type').size()

    grouped_by_fuel_type.plot(kind='bar', color='skyblue', edgecolor='black')

    plt.xlabel('Fuel Type')
    plt.ylabel('Number of vehicles')
    plt.title('Number of vehicles by fuel type')

    plt.show()

def e(data):
    avg_co2_by_cylinders = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=avg_co2_by_cylinders.index, y=avg_co2_by_cylinders.values, color='skyblue')

    plt.xlabel('Number of cylinders')
    plt.ylabel('Average CO2 emissions (g/km)')
    plt.title('Average CO2 emissions by number of cylinders')

    plt.show()

e(data)