""" 
The idea behind this ML project is to build a model that will classify how much 
loan the user can take. It is based on the user’s marital status, education, number 
of dependents, and employments. We can build a linear model for this project.
"""

# Importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Load dataset
dataset_train = pd.DataFrame(pd.read_csv('loan_train.csv'))
dataset_test = pd.DataFrame(pd.read_csv('loan_test.csv'))

# Split datasets
x_train = dataset_train.iloc[:, 1:12].values
y_train = dataset_train.iloc[:, 12].values

x_test = dataset_test.iloc[:, 1:12].values

# Encoding dataset
# Change array of obj to array of str
def obj_to_str(data):
    label_encoder = LabelEncoder()
    x_arr_str = []
    for el in data:
        x_arr_str.append(el)
    x_arr_str = np.array(x_arr_str)
    new_col = label_encoder.fit_transform(x_arr_str)
    
    return new_col

# Gender
x_train[:, 0] = obj_to_str(x_train[:, 0])

# isMarried
x_train[:, 1] = obj_to_str(x_train[:, 1])



# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = 'nan', strategy = 'mean')
imputer = imputer.fit(x_train[:, 0])
X[:, 0] = imputer.transform(X[:, 0])

# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
print(type(values))
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
print(integer_encoded)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
