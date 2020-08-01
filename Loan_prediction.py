""" 
The idea behind this ML project is to build a model that will classify how much 
loan the user can take. It is based on the user’s marital status, education, number 
of dependents, and employments. We can build a linear model for this project.
"""

# Importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Load dataset
dataset_train = pd.DataFrame(pd.read_csv('loan_train.csv'))
dataset_test = pd.DataFrame(pd.read_csv('loan_test.csv'))

# Split training datasets
x_train = dataset_train.iloc[:, 1:12]
y_train = dataset_train.iloc[:, 12]

# Split testing dataset
x_test = dataset_test.iloc[:, 1:12]

# Using get_dummies to encoding some columns
columns_encoding = ['Gender',
                    'Married',
                    'Education', 
                    'Self_Employed', 
                    'Property_Area']

x_train = pd.get_dummies(
    x_train,
    columns = columns_encoding) 

x_test = pd.get_dummies(
    x_test, 
    columns = columns_encoding)

# y_train encoding
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)


# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    # most_frequent
    # mean
    
missing_data_cols = ['Dependents', 
                     'LoanAmount', 
                     'Loan_Amount_Term', 
                     'Credit_History']

for index, val in enumerate(x_train['Dependents']):
    if val == '3+':
        x_train['Dependents'][index] = 3.0

for index, val in enumerate(x_test['Dependents']):
    if val == '3+':
        x_test['Dependents'][index] = 3.0

# Change str type to float
x_train['Dependents'] = x_train['Dependents'].astype(float)
x_test['Dependents'] = x_test['Dependents'].astype(float)

for col in missing_data_cols:
    x_train[col] = imputer.fit_transform(
        np.array(
            x_train[col].values.reshape(-1,1))
        )[:, 0]
    x_test[col] = imputer.fit_transform(
        np.array(
            x_test[col].values.reshape(-1,1))
        )[:, 0]
    
    
# Feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Find best model to prediction
# Import Models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
   
names = []
results = []

for name, model in models:
    kfold = StratifiedKFold(
        n_splits=10,
        random_state=1,
        shuffle=True)
    
    cv_results = cross_val_score(
        model,
        x_train,
        y_train,
        cv = kfold,
        scoring='accuracy')
    
    names.append(name)
    results.append(cv_results)
    print(name, cv_results.mean(), cv_results.std())
    
    
model = LinearDiscriminantAnalysis()
model.fit(x_train, y_train)
predict = model.predict(x_test)