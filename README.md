#Data wragling and preparation 

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

data = pd.read_excel("Ergometry.xlsx",sheet_name='Tabelle1') # Read excel file

y_varaible = data.pop('Test(Watt)') # Outcome variable Y ('Test(Watt)' column form excel file 
y_varaible = y_varaible[:650] # Taking only 650 rows
data=data[:650] # Taking only 650 rows
data.fillna(data.mean(), inplace=True) # Replacing all NA values with mean value
data.head()
data.describe()

#Random Forest Classifier



