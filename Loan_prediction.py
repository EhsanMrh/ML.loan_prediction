""" 
The idea behind this ML project is to build a model that will classify how much 
loan the user can take. It is based on the user’s marital status, education, number 
of dependents, and employments. We can build a linear model for this project.
"""

# Importing libraries
import pandas as pd
import numpy as np


# Load dataset
dataset_train = pd.DataFrame(pd.read_csv('loan_train.csv'))
dataset_test = pd.DataFrame(pd.read_csv('loan_test.csv'))

# Using get_dummies to encoding some columns
dataset_train_dummies = pd.get_dummies(dataset_train, 
                                 columns = [
                                     'Gender',
                                     'Married' ,
                                     'Education', 
                                     'Self_Employed', 
                                     'Property_Area',
                                     'Loan_Status'])
 
dataset_test_dummies = pd.get_dummies(dataset_train, 
                                 columns = [
                                     'Gender',
                                     'Married' ,
                                     'Education', 
                                     'Self_Employed', 
                                     'Property_Area'])


# Split datasets
x_train = dataset_train_dummies.iloc[:, 1:18].values
y_train = dataset_train_dummies.iloc[:, 18:20].values

x_test = dataset_test.iloc[:, 1:12].values




# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = 'nan', strategy = 'mean')
imputer = imputer.fit(x_train[:, 0])
X[:, 0] = imputer.transform(X[:, 0])

# get_dummies
t_dummies = pd.get_dummies(x_train[:, 0]) 
x_train[:, 1] = t_dummies



   

# =============================================================================
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# # Encoding dataset
# # Change array of obj to array of str
# label_encoder = LabelEncoder()
# def obj_to_str(column):
#     x_arr_str = []
#     for el in column:
#         x_arr_str.append(el)
#     x_arr_str = np.array(x_arr_str)
#     new_col = label_encoder.fit_transform(x_arr_str)
#     
#     return new_col
# 
# # Gender
# x_train[:, 0] = obj_to_str(x_train[:, 0])
# # isMarried
# x_train[:, 1] = obj_to_str(x_train[:, 1])
# # Education
# x_train[:, 3] = obj_to_str(x_train[:, 3])
# # Self_Employed
# x_train[:, 4] = obj_to_str(x_train[:, 4])
# # Property_Area
# x_train[:, 10] = obj_to_str(x_train[:, 10])
# 
# # OneHotEncoder    
# # creating one hot encoder object with categorical feature 0 
# # indicating the first column 
# from sklearn.compose import ColumnTransformer 
#    
# # creating one hot encoder object with categorical feature 0 
# # indicating the first column 
# onehotencoder = OneHotEncoder() 
# 
# columnTransformer = ColumnTransformer([('encoder', 
#                                         onehotencoder, 
#                                         [0])], 
#                                       remainder='passthrough') 
# 
# test = np.array(columnTransformer.fit_transform(x_train), dtype = np.str) 
# =============================================================================