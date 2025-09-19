# Support for efficient numerical computation
import numpy as np
# Library that supports dataframes
import pandas as pd
# Functions for machine learning
from sklearn.model_selection import train_test_split
# Entire preprocessing module: scaling, transforming and wrangling data
from sklearn import preprocessing
# Import random forest model
from sklearn.ensemble import RandomForestRegressor
# Import tools to perform cross-validation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
# Import evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score
# Persist model for future use
from sklearn.externals import joblib

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep = ';')

# Separate target from training features
y = data.quality
x = data.drop('quality', axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123,
                                                    stratify = y)
x_train_scaled = preprocessing.scale(x_train)
scaler = preprocessing.StandardScaler().fit(x_train)
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators = 100))

# Declare hyperparameters to tune
hyperparameters = {'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
                   'randomforestregressor__max_depth': [None, 5, 3, 1]}

clf = GridSearchCV(pipeline, hyperparameters, cv = 10)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)


print(data.head())
print(data.shape)
print(data.describe())
print(x_train_scaled)
print(x_train_scaled.mean(axis = 0))
print(x_train_scaled.std(axis = 0))
print(pipeline.get_params())
print(clf.best_params_)
print(clf.refit)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

# 10. Save model for future use
joblib.dump(clf, 'rf_regressor.pkl')
# To load: clf2 = joblib.load('rf_regressor.pkl')