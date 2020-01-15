# Data wragling and preparation 

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

data = pd.read_excel("Ergometry.xlsx",sheet_name='Tabelle1') # Read excel file

y_variable = data.pop('Test(Watt)') # Outcome variable Y ('Test(Watt)' column form excel file 
y_variable = y_variable[:650] # Taking only 650 rows
data=data[:650] # Taking only 650 rows
data.fillna(data.mean(), inplace=True) # Replacing all NA values with mean value
data.head()
data.describe()

# 1. Random Forest Classifier

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

X_train, X_test, y_train, y_test = train_test_split(data, y_variable, test_size = 0.25, random_state = 42) #Train and test model
model = RandomForestClassifier(n_estimators=100,criterion='gini', oob_score=True, random_state=42)
model.fit(X_train, y_train)

model.oob_score_
y_oob = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_oob)) # Returns accuracy of model
print("R2",model.oob_score_) # returns R squared

# Visualizing Importantce of Features

model.feature_importances_
feature_imp = pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)
feature_imp
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

# Predicting maximal Watt by entering required parameters

age = input("Please enter the Age: ")
age = int(age)
bw = input ("Please ebter the body weight: ")
bw = int(bw)
bh = input("Please enter the body height: ")
bh = int(bh)
bmi = input("Please enter the BMI: ")
bmi = int(bmi)

pred_RFC = model.predict([[age,bw,bh,bmi]])
int(pred_RFC)

# 2. Random Forest Regressor

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
X_train, X_test, y_train, y_test = train_test_split(data, y_variable, test_size = 0.25, random_state = 42)
model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import r2_score
model.oob_score_ 
y_oob = model.predict(X_test)
print("R2",r2_score(y_test, y_oob)) # returns R squared

# Visualizing Importantce of Features

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
model.feature_importances_
feature_imp = pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)
feature_imp
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

# Predicting maximal Watt by entering required parameters

age = input("Please enter the Age:")
age = int(age)
bw = input ("Please ebter the body weight:")
bw = int(bw)
bh = input("Please enter the body height:")
bh = int(bh)
bmi = input("Please enter the BMI:")
bmi = int(bmi)

pred_RFR = model.predict([[age,bw,bh,bmi]])
int(pred_RFR)

# 3. Multinomial Logistic Regression

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test=train_test_split(data,y_variable,test_size=0.25,random_state=42)

logreg = LogisticRegression(random_state=42) # instantiate the model (using the default parameters)
logreg = LogisticRegression(solver='lbfgs',multi_class='multinomial')
logreg.fit(X_train,y_train) # fit the model with data

# Visualizing Importantce of Features

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
model.feature_importances_
feature_imp = pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)
feature_imp
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

# Predicting maximal Watt by entering required parameters

y_pred=logreg.predict(X_test)
logreg.score(X_test,y_test)
age = input("Please enter the Age: ")
age = int(age)
bw = input ("Please ebter the body weight: ")
bw = int(bw)
bh = input("Please enter the body height: ")
bh = int(bh)
bmi = input("Please enter the BMI: ")
bmi = int(bmi)
pred = logreg.predict([[age,bw,bh,bmi]])
int(pred)
