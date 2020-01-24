#- Exploration des données
# Importing Data Science Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
#%matplotlib inline
from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score,confusion_matrix,precision_recall_curve,roc_curve
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.neighbors  import KNeighborsClassifier
from sklearn import tree

#####################
# Reading Csv file
path='./data/'
dftrain = pd.read_csv(path+'carInsurance_train.csv', encoding='utf8', sep=',')
dftrain.set_index('Id')
dftest = pd.read_csv(path+'carInsurance_test.csv', encoding='utf8', sep=',')
dftest.set_index('Id')

print(dftrain.shape)
print(dftrain.shape)
print(dftrain.info())
print(dftrain.columns)
print( dftrain.describe() )
print( dftest.describe() )
print( dftrain.dtypes )
print( dftrain.select_dtypes(include=['object']).head() )
print( dftrain.describe(include=['O']) )
print( dftrain.select_dtypes(include=['int64','float64']).head() )
print( dftrain.isnull().sum() )
print( dftest.isnull().sum() )
numeriquestrain =  dftrain.select_dtypes(include=['int64','float64'])
del numeriquestrain['Id']
numeriquestrain.hist(figsize=(12,12))

print( '*************************' )
numeriquestest =  dftest.select_dtypes(include=['int64','float64'])
del numeriquestest['Id']
numeriquestest.hist(figsize=(12,12))
print( '*******************' )
sns.pairplot(dftrain)
plt.savefig("DataSet.png")
plt.show()
print(  )















