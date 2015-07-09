import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

loan_csv_file =  '../data/LoanStats3a.csv'

def p2f(x):
    if(x.endswith("%")):
        return float(x.strip("%"))/100.
    else:
        print x

def ytof(x):
    if(x.endswith('10+ years')):
        return 10.
    elif(x.endswith('< 1 year')):
            return 0.
    elif(x.endswith('1 year')):
        return float(x.strip(" year"))
    elif(x.endswith('years')):
        return float(x.strip(" years"))



loan_grade  = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
# loan_status = {"Fully Paid": 4, "Current": 3, "In Grace Period": 2, "Late (31-120 days)": 1, "Charged Off": 0}
loan_status = {"Fully Paid": 1, "Current": 1, "In Grace Period": 1, "Late (31-120 days)": 0, "Late (16-30 days)": 0, "Charged Off": 0, "Default": 0}
# loan_status = {"Fully Paid": 1, "Charged Off": 0}
ho          =  {"OWN" : 1, "MORTGAGE" : 1, "RENT" : 0}

df = pd.read_csv(loan_csv_file, header = 1, converters =
{"grade": loan_grade.get, "int_rate" : p2f, "loan_status" : loan_status.get, "home_ownership" : ho.get, "emp_length":ytof, 'revol_util': p2f})
# df.describe()

# Remove rows where loan_status is not one of the 5 above
df = df[np.isfinite(df['loan_status'])]
df = df.fillnan(0)
 # Remove unwanted columns
 # X_data = df[['home_ownership', 'annual_inc', 'dti','delinq_2yrs',
 # 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',
 # 'open_acc','pub_rec','revol_bal','total_acc', 'emp_length']]
 #
 # y_data = df['loan_status']
 # d_trunc =  df[['home_ownership', 'annual_inc', 'dti','delinq_2yrs',
 # 'inq_last_6mths', 'open_acc','pub_rec','revol_bal', 'revol_util', 'total_acc', 'emp_length', 'loan_status']]
d_trunc =  df[['home_ownership', 'annual_inc', 'dti','delinq_2yrs',
  'inq_last_6mths', 'open_acc','pub_rec','revol_bal', 'revol_util', 'total_acc', 'emp_length']]
d_trunc['home_ownership'] = d_trunc['home_ownership'].fillna(0)


d_scaled = pd.DataFrame(scaler.fit_transform(d_trunc), columns=d_trunc.columns)
d_scaled['loan_status'] = df['loan_status']

# d_trunc['is_train'] = np.random.uniform(0, 1, len(d_trunc)) <= .75
# train, test = d_trunc[d_trunc['is_train']==True], d_trunc[d_trunc['is_train']==False]
# features = d_trunc.columns[:11]
d_scaled['is_train'] = np.random.uniform(0, 1, len(d_scaled)) <= .75
train, test = d_scaled[d_scaled['is_train']==True], d_scaled[d_scaled['is_train']==False]
features = d_scaled.columns[:11]

y = train['loan_status']
testy = test['loan_status']
clf.fit(train[features], y)

preds = clf.predict(test[features])
pd.crosstab(testy, preds, rownames=['actual'], colnames=['preds'])
