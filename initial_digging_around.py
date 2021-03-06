import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
# Custom utils
from converters import p2f, ytof, str2words
from plotters import get_results, plot_roc

loan_csv_file =  '../data/LoanStats3a_securev1.csv'

# Coonvert the formatting from text to numbers
loan_grade  = {"A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1, "G":0}
loan_status = {"Fully Paid": 1, "Current": 555, "In Grace Period": 555, "Late (31-120 days)": 0, "Late (16-30 days)": 0, "Charged Off": 0, "Default": 0}
ho          =  {"OWN" : 1, "MORTGAGE" : 1, "RENT" : 0}

# Read the data file
df = pd.read_csv(loan_csv_file, header = 1, converters ={"grade": loan_grade.get,
"int_rate" : p2f, "loan_status" : loan_status.get, "home_ownership" : ho.get, "emp_length":ytof, 'revol_util': p2f, 'desc' : str2words}, low_memory=False)

select_features = ['fico_range_high','loan_amnt', 'int_rate', 'grade', 'home_ownership', 'annual_inc', 'dti','delinq_2yrs',
'inq_last_6mths', 'open_acc','pub_rec','revol_bal', 'revol_util', 'total_acc', 'emp_length', 'desc', 'loan_status']

df = df[df.loan_status != 555]
df = df.fillna(0)

# Remove unwanted columns
d_trunc =  df[select_features]
d_trunc['home_ownership'].fillna(0)

x = 'annual_inc'; y = 'grade'
# x = 'int_rate'; y = 'grade'
# x = 'loan_status'; y = 'revol_bal'
# x = 'fico_range_high'; y = 'int_rate'
# x = 'fico_range_high'; y = 'annual_inc' # Interesting, dont see a trend!
# x = 'fico_range_high'; y = 'delinq_2yrs'
pl.scatter(d_trunc[x],d_trunc[y])
pl.xlabel(x); pl.ylabel(y);
pl.show()
