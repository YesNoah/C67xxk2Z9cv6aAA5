from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def pred_grad_boost(X_test, gbc):
    y_pred_gbc=gbc.predict(X_test)
    return y_pred_gbc

def score_grad_boost(y_test, y_pred_gbc, gbc):
    confusion_gbc=confusion_matrix(y_test,y_pred_gbc)
    plt.figure(figsize=(8,8))
    sns.heatmap(confusion_gbc,annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    print(classification_report(y_test,y_pred_gbc))
    print(f'\nBest Parameter: {gbc.best_params_}\n')