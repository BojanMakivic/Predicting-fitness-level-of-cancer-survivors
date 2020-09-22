# Predicting maximal work capacity of cancer survivors
## Preface
I can only recommend you the book [**Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow**](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) written by *Aurélien Géron* which served me as a great help while I've been working on this repository.

<img src="https://images-na.ssl-images-amazon.com/images/I/51aqYc1QyrL._SX379_BO1,204,203,200_.jpg" title="book" width="150" />

## Introduction
After finishing with surgery and drug therapy, the cancer patients should undergo physical and sport therapy in order to regain their reduced physical capacity (strength, endurance, flexibility and etc.). In order to tailor optimal training's plan for every patient individualy, we enter the process of information geathering in terms if person's fitness level. One quite common test that is performed in the clinical setup is cycle ergometry test which is suitable for cardiorespiratory and aerobic fitnees assessment. 
Adventages of this test are many, ranging from reliability and validity of the test to accuracy and easy interpretation. On the other hand disadvantages are mostly related to organizational, time-management and cost aspects.

## Data
The dataset include age, gender, body weight, body height and highest worload (Watt) achieved during cycle ergometer test.
## Models
Three models (Linear Regression, Support Vector Regressor and Random Forest Regressor) are used to predict maximal Watt performace which can be obtained during maximal cycle ergometry test. Four attributes (Age, body weight, body height and body mass index (BMI)) are used to predict outcome variable (WATTmax). 
