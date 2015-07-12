import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# Custom utils
from converters import p2f, ytof, str2words, term2f
from plotters import get_results, plot_roc
# import pickle

# loan_csv_file =  '../data/LoanStats3a.csv'
loan_csv_file =  '../data/LoanStats3a_securev1.csv'

# Coonvert the formatting from text to numbers
loan_grade  = {"A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1, "G":0}
# loan_status = {"Fully Paid": 1, "Current": 555, "In Grace Period": 555, "Late (31-120 days)": 555, "Late (16-30 days)": 555, "Charged Off": 0, "Default": 0}
loan_status = {"Fully Paid": 0, "Current": 555, "In Grace Period": 555, "Late (31-120 days)": 555, "Late (16-30 days)": 555, "Charged Off": 1, "Default":  555 }

ho          =  {"OWN" : 1, "MORTGAGE" : 1, "RENT" : 0}

df = pd.read_csv(loan_csv_file, header = 1, converters ={"grade": loan_grade.get,
    "int_rate" : p2f, "loan_status" : loan_status.get, "home_ownership" : ho.get, "emp_length":ytof, 'revol_util': p2f, 'desc' : str2words, 'term' : term2f}, low_memory=False)

# select_features = ['fico_range_high', 'term', 'loan_amnt','int_rate', 'grade', 'home_ownership', 'annual_inc', 'dti','delinq_2yrs',
# 'inq_last_6mths', 'open_acc','pub_rec','revol_bal', 'revol_util', 'total_acc', 'emp_length', 'desc', 'loan_status']

select_features = ['fico_range_high', 'term', 'loan_amnt','int_rate', 'grade', 'annual_inc', 'dti','delinq_2yrs',
'inq_last_6mths', 'open_acc','pub_rec','revol_bal', 'revol_util', 'total_acc', 'emp_length', 'desc']
select_label = ['loan_status']

# pub_rec, delinq_2yrs: mostly all 0s
print("Using following {0} features: {1}".format(len(select_features), select_features))

# Remove rows where loan_status is not one of the 5 above
df = df[df.loan_status != 555]
# df = df.fillna(0)
df = df[df.emp_length != 555]
df = df[np.isfinite(df['revol_util'])]

# Remove unwanted columns
# d_trunc =  df[select_features].astype(float)

features = df[select_features].astype(float)
labels = df[select_label]

# standardize
features = features[select_features].apply(lambda x: MinMaxScaler().fit_transform(x))

# features.to_csv('../data/features.csv')
# labels.to_csv('../data/labels.csv')

# Create train/test partition
features['is_train'] = np.random.uniform(0, 1, len(features)) <= .75
Xtrain, Xtest = features[features['is_train']==True], features[features['is_train']==False]
ytrain, ytest = labels[features['is_train']==True], labels[features['is_train']==False]
print("Train shape = {0}".format(Xtrain.shape))
print("Test shape = {0}".format(Xtest.shape))
# print("-------------------------------------------------")
# # print("Total data: {0}, train data: {1} [{2}%], test data: {3}[{4}%]".format(d_trunc.shape[0],
# # train.shape[0], np.round(train.shape[0]/float(d_trunc.shape[0])*100), test.shape[0], np.round(test.shape[0]/float(d_trunc.shape[0])*100)))
#
# # features = d_trunc.columns[:len(select_features)-1]
# # y = train['loan_status']
# # testy = test['loan_status']
#
# # Generate a weigtage for balancing data
# # weight = np.array([5 if i == 0 else 1 for i in y])
# # print(train[features])
# # Traing a random forest classifier
# clf = RandomForestClassifier(n_jobs = 2, class_weight = {1: 4, 0: 1}, n_estimators = 11, max_depth=None)
# for i in range(1,20,2):
#     print("***{0}***".format(i))
#     # clf = RandomForestClassifier(n_estimators = 11, class_weight = {1: 1, 0: 3}, max_depth=None)
#     clf = RandomForestClassifier(n_estimators = i, class_weight = {1: 1, 0: 3}, max_depth=None)
#     clf.fit(Xtrain[select_features], np.squeeze(ytrain))
#     preds = clf.predict(Xtest[select_features])
#     print("-------------------------------------------------")
#     fpr, tpr, roc_auc = get_results(np.squeeze(ytest), preds)

clf = RandomForestClassifier(n_estimators = 11, class_weight = {1: 1, 0: 3}, max_depth=None)
clf.fit(Xtrain[select_features], np.squeeze(ytrain))
preds = clf.predict(Xtest[select_features])
print("-------------------------------------------------")
fpr, tpr, roc_auc = get_results(np.squeeze(ytest), preds)
pl.figure()
plot_roc(fpr, tpr, roc_auc)
#
# # Traing a logistic regression classifier
# lore = LogisticRegression(C=10, penalty = 'l1', tol = 0.01)
lore = LogisticRegression(C=11, class_weight = 'auto')

lore.fit(Xtrain[select_features], np.squeeze(ytrain))
lorepreds = lore.predict(Xtest[select_features])
fpr, tpr, roc_auc = get_results(np.squeeze(ytest), lorepreds)
plot_roc(fpr, tpr, roc_auc)
pl.show()

# from sklearn.metrics import accuracy_score, roc_auc_score
# for i in range(1, 30, 1):
#
#     lore = LogisticRegression(C=i, class_weight = 'auto')
#     lore.fit(Xtrain[select_features], np.squeeze(ytrain))
#     lorepreds = lore.predict(Xtest[select_features])
#     # print("-------------------------------------------------")
#     # fpr, tpr, roc_auc = get_results(np.squeeze(ytest), lorepreds)
#     print("C: {0}, AUC: {1}".format(i, roc_auc_score(np.squeeze(ytest), lorepreds)))
    # plot_roc(fpr, tpr, roc_auc)
    # pl.show()
#
# # Nearest neighbor
# knn = KNeighborsClassifier(n_neighbors = 1)
# knn.fit(Xtrain[select_features], np.squeeze(ytrain))
# knnpreds = knn.predict(Xtest[select_features])
# print("-------------------------------------------------")
# fpr, tpr, roc_auc = get_results(np.squeeze(ytest), knnpreds)
# plot_roc(fpr, tpr, roc_auc)
# pl.show()
# from sklearn.metrics import accuracy_score, roc_auc_score
# for n in range(1, 50, 2):
#     knn = KNeighborsClassifier(n_neighbors = n)
#     knn.fit(Xtrain[select_features], np.squeeze(ytrain))
#     knnpreds = knn.predict(Xtest[select_features])
#     print("-------------------------------------------------")
#     print("N = {0}".format(n))
#     print("AUC: {0}".format(roc_auc_score(np.squeeze(ytest), knnpreds)))
#     print("-------------------------------------------------")
