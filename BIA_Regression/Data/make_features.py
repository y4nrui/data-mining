import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as pyplot
from sklearn import preprocessing

train_df = pd.read_csv('./train.csv')
train_df['SalePrice'] = train_df['SalePrice'].apply(lambda x: math.log(x))


# FEATURE SELECTION based on missing data, relevance, balance of the data, and repeated columns
insufficient_features = ['Alley', 'Fence', 'MiscFeature', 'PoolQC', 
                     'FireplaceQu', 'LotFrontage']
imbalanced_features = ['3SsnPorch','LandSlope', 'LowQualFinSF',
                   'PoolArea', 'RoofMatl', 'Street', 'Utilities']
irrelevant_features = ['Condition2','BsmtHalfBath','YrSold','Id']
train_df.drop(imbalanced_features + insufficient_features + irrelevant_features, axis =1, inplace = True)

# LOW CORRELATIONS
test = train_df.corr()
print(test)

low_corr = ['MiscVal', 'BsmtFinSF2']
high_corr = ['GarageCars', 'GarageYrBlt', 'TotalRmsAbvGrd', 'TotalBsmtSF']
train_df.drop(low_corr, axis = 1, inplace= True)
print('hello world')

# Dealing with missing data
train_df.update(train_df[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].fillna('No Basement'))
train_df.update(train_df[['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']].fillna('No Garage'))
train_df['Electrical'] = train_df['Electrical'].fillna('SBrkr')
train_df['MasVnrType'] = train_df['MasVnrType'].fillna('None')
train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(0)

train_df.drop('GarageYrBlt', axis =1, inplace=True)

# PREPROCESSING
train_df = train_df.astype({'MSSubClass': 'object', 'MoSold': 'object'})
y = train_df['SalePrice']
train_df.drop('SalePrice', axis = 1, inplace= True)
quant_var = list(train_df.columns[train_df.dtypes != 'object'])
standardized_vals = preprocessing.StandardScaler().fit_transform(train_df[quant_var])
standardized_vals = pd.DataFrame(standardized_vals, columns = quant_var)

# One hot encoding
qual_var = list(train_df.columns[train_df.dtypes == 'object'])
qual_var_one_hot = pd.get_dummies(train_df[qual_var], prefix = qual_var, columns =  qual_var)
qual_var_label = train_df[qual_var].apply(preprocessing.LabelEncoder().fit_transform)

# Different Datasets
one_hot_final = pd.merge(standardized_vals, qual_var_one_hot, left_index = True, right_index = True)
one_hot_final = pd.merge(y, one_hot_final, left_index= True, right_index = True)
# print(one_hot_final)
one_hot_final.to_csv('one_hot_df.csv')

label_final = pd.merge(standardized_vals, qual_var_label, left_index = True, right_index = True)
label_final = pd.merge(y, label_final, left_index= True, right_index = True)
label_final.to_csv('label_df.csv')