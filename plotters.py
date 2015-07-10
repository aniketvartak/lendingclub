import pandas as pd
from sklearn.metrics import *
import matplotlib.pyplot as pl

def get_results(testy, preds):
    print(pd.crosstab(testy, preds, rownames=['actual'], colnames=['preds']))
    print("Precision = {0}".format(round(precision_score(testy, preds), 4)))
    print("Recall = {0}".format(round(recall_score(testy, preds), 4)))
    print("Accuracy = {0}".format(round(accuracy_score(testy, preds), 4)))
    print("F1-score = {0}".format(round(f1_score(testy, preds), 4)))
    print("AUC: {0}".format(round(roc_auc_score(testy, preds), 4)))
    fpr, tpr, _ = roc_curve(testy, preds)
    roc_auc     = auc(fpr, tpr)
    return fpr,tpr,roc_auc

def plot_roc(fpr, tpr, roc_auc):
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
