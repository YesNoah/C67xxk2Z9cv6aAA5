import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import statsmodels.api as sm
import scipy.stats as stats
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

def train_grad_boost(X_train, y_train):
    param_grid = {
        'learning_rate' : [.475],
        'n_estimators': [300],
    }

    gbc=GridSearchCV(GradientBoostingClassifier(),param_grid,cv=5, verbose = 1000)
    gbc.fit(X_train,y_train)
    return(gbc)
