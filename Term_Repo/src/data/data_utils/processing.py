from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy.stats as stats
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import statsmodels.api as sm
import scipy.stats as stats
import sklearn

def prepro(df):
    # replace binary categorical features
    df['default'].replace(['no', 'yes'],
                        [0, 1], inplace=True)
    df['housing'].replace(['no', 'yes'],
                        [0, 1], inplace=True)
    df['loan'].replace(['no', 'yes'],
                        [0, 1], inplace=True)
    df['y'].replace(['no', 'yes'],
                        [0, 1], inplace=True)
    df['month'].replace(['jan', 'feb', 'mar', 'apr', 'may', 'jun','jul','aug', 'sep','oct', 'nov', 'dec'],
                        [1,2,3,4,5,6,7,8,9,10,11,12], inplace=True)
    # encode multi label categories with dummy variables
    job = df.job
    jobencode = pd.get_dummies(job)
    marital = df.marital
    maritalencode = pd.get_dummies(marital)
    education = df.education
    educationencode = pd.get_dummies(education)
    contact = df.contact
    contactencode = pd.get_dummies(contact)
    # append new encoded columns and drop old categorical columns
    dft = pd.concat([df,maritalencode],
                axis = 1)
    dft = pd.concat([dft,educationencode],
                axis=1)
    dft = pd.concat([dft,contactencode],
                axis=1)
    dft = pd.concat([dft,jobencode],
                 axis=1)
    dft = dft.drop(['job','marital','education','contact'], axis = 1)

    # pop y to first column
    column_to_move = dft.pop("y")

    # insert column with insert(location, column_name, column_value)

    dft.insert(0, "y", column_to_move)

    sliced_features = dft.iloc[:,1:]
    sliced_labels = dft['y']
    sliced_features.head()

    # feature selection
    fs = SelectKBest(score_func=f_classif, k=25)
    X_selected = fs.fit_transform(sliced_features, sliced_labels)

    #Checking which features where selected
    filter = fs.get_support()
    feat = np.array(sliced_features.columns)
    print('Total Features ', feat)
    print('Selected Features for training ',feat[filter])

    #remove outliers
    z = np.abs(stats.zscore(X_selected))
    inputs = X_selected[(z<3).all(axis=1)]
    target = sliced_labels[(z<3).all(axis=1)]

    #scale inputs
    sc = StandardScaler()
    inputs = sc.fit_transform(inputs)

    #Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(inputs, target, random_state=1234)

    #Balance Dataset
    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)
    X_test, y_test = sm.fit_resample(X_test, y_test)
    print(X_train.shape)
    print(X_test.shape)

    #return processed dataframes
    return(X_train, X_test, y_train, y_test)