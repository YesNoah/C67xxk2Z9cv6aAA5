import os
os.getcwd()
import importlib
import data
importlib.reload(data)
from data import make_dataset
importlib.reload(make_dataset)
from data import data_utils
importlib.reload(data_utils)

X_train, X_test, y_train, y_test = make_dataset.main(path = r'O:\Term_Repo\data\raw', 
                                                    filename = "term-deposit-marketing-2020.csv", 
                                                    outputfilepath='O:\Term_Repo\data\processed\df.csv'
                                                    )
X_train


import models
import models.train_model as train
gbc = train.train_grad_boost(X_train, y_train)

import models.predict_model as pred
importlib.reload(models.predict_model)
y_pred_gbc = pred.pred_grad_boost(X_test, gbc)

pred.score_grad_boost(y_test, y_pred_gbc, gbc)
