import pandas as pd
import numpy as np

data = pd.read_csv('data_C02_emission.csv')
new_data = data.groupby('Cylinders')

print(new_data.count())
print(new_data.size())
print(new_data.sum())