import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

one_hot_df = pd.read_csv('Data/one_hot_df.csv')
label_df = pd.read_csv('Data/label_df.csv')
y = one_hot_df['SalePrice']
X = one_hot_df.drop(['SalePrice'], axis =1)

# MULTIVARIATE LINEAR REGRESSION
model = LinearRegression()
metrics = cross_validate(model, X,y,cv = 5, scoring = ('neg_root_mean_squared_error'))
lr_cv = -metrics['test_score'].mean()
print('SCORE FOR MULTIVARIATE LR')
print(lr_cv)

#RIDGE REGRESSION
param_list = []
start = 10
for i in range(30):
    param_list.append(start)
    start = start + 0.25

parameters = {'alpha': param_list} 
rr = Ridge()
metrics = cross_validate(rr, X, y, cv = 5, scoring = ('neg_root_mean_squared_error'))
print('SCORE FOR UNTUNED RR')
print(-metrics['test_score'].mean())

#TUNED RR
tuned_rr = GridSearchCV(rr, parameters, scoring = 'neg_root_mean_squared_error', cv = 5)
tuned_rr.fit(X,y)
print('SCORE FOR TUNED RR')
print(-tuned_rr.best_score_)
print(tuned_rr.best_params_)
print(tuned_rr.best_estimator_)

#LASSO REGRESSION TUNED
param_list = []
start = 0.0001
for i in range(30):
    param_list.append(start)
    start = start*2
parameters2 = {'alpha': param_list}
lasso = Lasso()
metrics = cross_validate(lasso, X,y, cv = 5, scoring = ('neg_root_mean_squared_error'))
tuned_lasso = GridSearchCV(lasso, parameters2, scoring = 'neg_root_mean_squared_error', cv = 5)
tuned_lasso.fit(X,y)
print('SCORE FOR LASSO')
print(-tuned_lasso.best_score_)
print(tuned_lasso.best_params_)
print(tuned_lasso.best_estimator_)

# HYPERPARAMETER TUNING OF RANDOM FOREST MODEL
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num =10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10,110,num =11)]
max_depth.append(None)
min_samples_split = [2,5,10]
min_samples_leaf = [1,2,4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 25, cv =2, verbose = 2, random_state = 42, n_jobs = -1, scoring = 'neg_root_mean_squared_error', bootstrap = True)
rf_random.fit(X, y)
print('THIS IS THE BEST SCORE')
print(-rf_random.best_score_)
print('THIS IS THE BEST PARAMS')
print(rf_random.best_params_)

param_grid = {
    'max_depth': [37,40,43],
    'min_samples_leaf': [1],
    'min_samples_split': [2],
    'max_features' : ['sqrt'],
    'n_estimators': [800],
    'bootstrap': [True]
}

rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs= -1, verbose = 2, scoring = 'neg_root_mean_squared_error')
grid_search.fit(X,y)
# print(grid_search.best_params_)
print('SCORE FOR RF')
print(-grid_search.best_score_)

y_rf = label_df['SalePrice']
X_rf = label_df.drop(['SalePrice'], axis = 1)
rfmodel = RandomForestRegressor (bootstrap = False, max_depth = 37, min_samples_leaf= 1, min_samples_split=2, max_features = 'sqrt', n_estimators = 675)
rfmetrics = cross_validate(rfmodel, X_rf,y_rf, cv = 5, scoring = ('neg_root_mean_squared_error'))
print('SCORE FOR RF')
print(-rfmetrics['test_score'].mean())