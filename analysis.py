import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
# Custom utils
from converters import p2f, ytof, str2words, term2f
from plotters import get_results, plot_roc

loan_csv_file =  '../data/LoanStats3a.csv'

# Coonvert the formatting from text to numbers
loan_grade  = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
loan_status = {"Fully Paid": 1, "Current": 555, "In Grace Period": 555, "Late (31-120 days)": 0, "Late (16-30 days)": 0, "Charged Off": 0, "Default": 0}
ho          =  {"OWN" : 1, "MORTGAGE" : 1, "RENT" : 0}

# Read the data file
df = pd.read_csv(loan_csv_file, header = 1, converters ={"grade": loan_grade.get,
"int_rate" : p2f, "loan_status" : loan_status.get, "home_ownership" : ho.get, "emp_length":ytof, 'revol_util': p2f, 'desc' : str2words, 'term' : term2f}, low_memory=False)

select_features = ['term', 'loan_amnt','int_rate', 'grade', 'home_ownership', 'annual_inc', 'dti','delinq_2yrs',
'inq_last_6mths', 'open_acc','pub_rec','revol_bal', 'revol_util', 'total_acc', 'emp_length', 'desc', 'loan_status']
print("Using following features: {0}".format(select_features))

# Remove rows where loan_status is not one of the 5 above
# df = df[np.isfinite(df['loan_status'])]
df = df[df.loan_status != 555]
df = df.fillna(0)

# Remove unwanted columns
d_trunc =  df[select_features]
d_trunc['home_ownership'].fillna(0)

# Scale the data in [0,1] range
d_trunc[select_features].apply(lambda x: MinMaxScaler().fit_transform(x))

# Create train/test partition
d_trunc['is_train'] = np.random.uniform(0, 1, len(d_trunc)) <= .75
train, test = d_trunc[d_trunc['is_train']==True], d_trunc[d_trunc['is_train']==False]
print("-------------------------------------------------")
print("Total data: {0}, train data: {1} [{2}%], test data: {3}[{4}%]".format(d_trunc.shape[0],
train.shape[0], np.round(train.shape[0]/float(d_trunc.shape[0])*100), test.shape[0], np.round(test.shape[0]/float(d_trunc.shape[0])*100)))

features = d_trunc.columns[:len(select_features)-1]
y = train['loan_status']
testy = test['loan_status']

# Generate a weigtage for balancing data
weight = np.array([5 if i == 0 else 1 for i in y])

# Traing a random forest classifier
clf = RandomForestClassifier(n_jobs = 2, class_weight = {1: 5, 0: 1})
clf.fit(train[features], y)
preds = clf.predict(test[features])
print("-------------------------------------------------")
fpr, tpr, roc_auc = get_results(testy, preds)
pl.figure()
plot_roc(fpr, tpr, roc_auc)

# Traing a logistic regression classifier
lore = linear_model.LogisticRegression(C=1e5, penalty = 'l1', tol = 0.01)
lore.fit(train[features], y)
lorepreds = lore.predict(test[features])
print("-------------------------------------------------")
fpr, tpr, roc_auc = get_results(testy, lorepreds)
plot_roc(fpr, tpr, roc_auc)
pl.show()
