# DATA PREPARATION AND LIBRARIES IMPORT

# Python ≥3.5 
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd

# Figures plot
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Figures dir save
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore warnings
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# GUI to import data file (xlsx)

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 300, height = 300, bg = 'lightsteelblue2', relief = 'raised')
canvas1.pack()

def getCSV ():
    global df
    
    import_file_path = filedialog.askopenfilename()
    df = pd.read_excel (import_file_path)
    df = pd.DataFrame(df)
    
browseButton_CSV = tk.Button(text="      Import XLSX File     ", command=getCSV, bg='green', fg='white', font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 150, window=browseButton_CSV)

root.mainloop()

df.info()

# Histogram of gender distribution
df['Gender'].hist(label=int,align='mid')

# Function to calculate bmi index
def bmi(bw,bh):
    return bw/((bh/100)*(bh/100))

# Add a new column with the name "BMI"
df['BMI']=bmi(df.iloc[:,2],df.iloc[:,3]) 

df.describe()

# Histogram distribution for each attribute

df.hist(bins=50, figsize=(20,15))
save_fig("attribute_histogram_plots")
plt.show()

# to make this notebook's output identical at every run
np.random.seed(42)


# PREPARING TRAINING AND TEST SET

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
train_labels = train_set["Wmax"].copy()       # Save a copy of label variable
train_set = train_set.drop("Wmax", axis=1)    # Drop label variable from training set

train_num = train_set.drop('Gender', axis=1)  # Keep only numerical variables

# Pipline to scale numerical variables and transform categorical variable (gender) into binary

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
        ('std_scaler', StandardScaler()),])
train_transf = pipeline.fit_transform(train_num)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

num_attribs = list(train_num)
cat_attribs = ["Gender"]

full_pipeline = ColumnTransformer([
        ("num", pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),])

train_prepared = full_pipeline.fit_transform(train_set) # Training set is ready for use

# SELECTING AND TRAINING THE MODEL
## Linear regression

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(train_prepared, train_labels)

# RMSE for linear model

from sklearn.metrics import mean_squared_error

watt_predictions = lin_reg.predict(train_prepared)
lin_mse = mean_squared_error(train_labels, watt_predictions)
lin_rmse = np.sqrt(lin_mse)
print("RMSE is: ",lin_rmse) # RMSE is:  32.46354580734144 -- We can see that the prediction error for linear model is 32.5 Watt 

## Support vector regressor (SVR)

from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(train_prepared, train_labels)

# RMSE for SVR model

watt_SVR_predictions = svm_reg.predict(train_prepared)
SVR_mse = mean_squared_error(train_labels, watt_SVR_predictions)
SVR_rmse = np.sqrt(SVR_mse)
print ("RMSE is: ",SVR_rmse) # RMSE is:  32.615674028371714 -- We can see that the prediction error for linear model is 32.6 Wat

## Random forest model

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
forest_reg.fit(train_prepared, train_labels)

# RMSE for random forest model

watt_RF_predictions = forest_reg.predict(train_prepared)
forest_mse = mean_squared_error(train_labels, watt_RF_predictions)
forest_rmse = np.sqrt(forest_mse)
print("RMSE is: ", forest_rmse) # RMSE is:  13.078592962697485 -- We can see that the prediction error for linear model is 13.1 Wat (until now the model with lowest prediction error)

## Fine tuning of models

'''The following code randomly splits the training set into 10 distinct subsets called folds, then it trains and evaluates the model 10 times, 
picking a different fold for evaluation every time and training on the other 9 folds. The result is an array containing the 10 evaluation scores:'''

from sklearn.model_selection import cross_val_score

# Function to display the score statistics

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
# Linear model

lin_scores = cross_val_score(lin_reg, train_prepared, train_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

''' Scores: [30.53455934 27.46985412 38.43562164 32.76486023 38.07501257 31.3886864531.97227142 31.11989247 32.31836557 30.94683766]
    Mean: 32.50259614886735
    Standard deviation: 3.184513995300587'''

# SVR

SVR_scores = cross_val_score(svm_reg, train_prepared, train_labels,
                                scoring="neg_mean_squared_error", cv=10)
SVR_rmse_scores = np.sqrt(-SVR_scores)
display_scores(SVR_rmse_scores)

''' Scores: [31.19126727 27.5297865  38.9271332  33.18049647 38.51193615 31.0531404231.92811072 30.98819848 32.00674191 30.15057619]
    Mean: 32.54673873052626
    Standard deviation: 3.387883572916082'''

# Random forest

forest_scores = cross_val_score(forest_reg, train_prepared, train_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

'''Scores: [34.41167315 29.82213183 40.39247186 34.37520891 36.24074161 31.6744174434.49259181 31.78694634 33.58427049 33.96558781]
   Mean: 34.07460412488601
   Standard deviation: 2.7261035077436997'''

# Return the table of cross-validation for all three models

scores = cross_val_score(lin_reg,  train_prepared, train_labels, scoring="neg_mean_squared_error", cv=10)
scores_rf = cross_val_score(forest_reg,  train_prepared, train_labels, scoring="neg_mean_squared_error", cv=10)
scores_svr = cross_val_score(svm_reg,  train_prepared, train_labels, scoring="neg_mean_squared_error", cv=10)
a = pd.Series(np.sqrt(-scores)).describe()
b = pd.Series(np.sqrt(-scores_svr)).describe()
c = pd.Series(np.sqrt(-scores_rf)).describe()
scores_df = pd.DataFrame({
    "lin_reg": a,
    "SVR":b,
    "forest_reg": c
    
})
scores_df

# Cross-validation table
'''	     lin_reg	forest_reg	SVR
count	10.000000	10.000000	10.000000
mean	32.502596	33.978674	32.546739
std	    3.356772	2.963467	3.571143
min	    27.469854	29.214386	27.529786
25%	    30.990101	32.274851	31.004434
50%	    31.680479	33.925521	31.559689
75%	    32.653237	34.738796	32.887058
max	    38.435622	40.232822	38.927133 '''
