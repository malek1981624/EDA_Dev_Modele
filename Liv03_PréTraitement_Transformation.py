# Importing Data Science Libraries
import datetime

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

# Lecture des données
path='data/'
train = pd.read_csv(path+'carInsurance_train.csv', encoding='utf8', sep=',')
test = pd.read_csv(path+'carInsurance_test.csv', encoding='utf8', sep=',')
# Split train and test
all=pd.concat([train,test],keys=('train','test'))
all.drop(['CarInsurance','Id'],axis=1,inplace=True)
print(all.shape)
all_df = all.copy()

##############################
# Remplacer les données manquantes de Outcome de la campagne précédente

all_df.loc[all_df['DaysPassed']==-1,'Outcome']='NoPrev'

# Remplacer les données manquantes de "Communication" avec "none"
all_df['Communication'].value_counts()
all_df['Communication'].fillna('other',inplace=True)

# Remplacer les données manquantes de "Education" avec la valeur la plus populaire avec type de job
all_df['Education'].value_counts()

# mappage job-education
edu_mode=[]
job_types = all_df.Job.value_counts().index
for job in job_types:
    mode = all_df[all_df.Job==job]['Education'].value_counts().nlargest(1).index
    edu_mode = np.append(edu_mode,mode)
edu_map=pd.Series(edu_mode,index=all_df.Job.value_counts().index)

# mappage de eductaion
for j in job_types:
    all_df.loc[(all_df['Education'].isnull()) & (all_df['Job']==j),'Education'] = edu_map.loc[edu_map.index==j][0]
all_df['Education'].fillna('other',inplace=True)

# remplacer les données manquantes de "Job" avec none
all_df['Job'].fillna('other',inplace=True)

################################
# Coder les valeurs numériques de l'age en cinq intervalles
all_df['AgeBand']=pd.cut(all_df['Age'],3)
print(all_df['AgeBand'].value_counts())

################################
all_df.loc[(all_df['Age']>=17) & (all_df['Age']<34),'AgeBin'] = 1
all_df.loc[(all_df['Age']>=34) & (all_df['Age']<49),'AgeBin'] = 2
all_df.loc[(all_df['Age']>=49) & (all_df['Age']<65),'AgeBin'] = 3
all_df.loc[(all_df['Age']>=65) & (all_df['Age']<80),'AgeBin'] = 4
all_df.loc[(all_df['Age']>=80) & (all_df['Age']<96),'AgeBin'] = 5
all_df['AgeBin'] = all_df['AgeBin'].astype(int)

#############################
# Coder les valeurs numériques de l'age en cinq intervalles
all_df['BalanceBand']=pd.cut(all_df['Balance'],5)
#############################
all_df.loc[(all_df['Balance']>=-3200) & (all_df['Balance']<17237),'BalanceBin'] = 1
all_df.loc[(all_df['Balance']>=17237) & (all_df['Balance']<37532),'BalanceBin'] = 2
all_df.loc[(all_df['Balance']>=37532) & (all_df['Balance']<57827),'BalanceBin'] = 3
all_df.loc[(all_df['Balance']>=57827) & (all_df['Balance']<78122),'BalanceBin'] = 4
all_df.loc[(all_df['Balance']>=78122) & (all_df['Balance']<98418),'BalanceBin'] = 5
all_df['BalanceBin'] = all_df['BalanceBin'].astype(int)

###########################3
# Supprimer le attributs en tranches:'AgeBand','BalanceBand','Age','Balance' et garder: AgeBin','BalanceBin'
all_df = all_df.drop(['AgeBand','BalanceBand','Age','Balance'],axis=1)
print(all_df.shape)
############################
# Mappage 'Education' en valeurs numeriques 0,1,2,3
all_df['Education'] = all_df['Education'].replace({'other':0,'primary':1,'secondary':2,'tertiary':3})
############################
import datetime
print("# Récupérer la longeur d'un appel téléphonique en minutes")
all_df['CallEnd'] = pd.to_datetime(all_df['CallEnd'])
all_df['CallStart'] = pd.to_datetime(all_df['CallStart'])
all_df['CallLength'] = ((all_df['CallEnd'] - all_df['CallStart'])/np.timedelta64(1,'m')).astype(float)
all_df['CallLenBand']=pd.cut(all_df['CallLength'],5)
print(all_df['CallLenBand'].value_counts())

###########################################
print("# Créer la longueur d'appel")
all_df.loc[(all_df['CallLength']>= 0) & (all_df['CallLength']<11),'CallLengthBin'] = 1
all_df.loc[(all_df['CallLength']>=11) & (all_df['CallLength']<22),'CallLengthBin'] = 2
all_df.loc[(all_df['CallLength']>=22) & (all_df['CallLength']<33),'CallLengthBin'] = 3
all_df.loc[(all_df['CallLength']>=33) & (all_df['CallLength']<44),'CallLengthBin'] = 4
all_df.loc[(all_df['CallLength']>=44) & (all_df['CallLength']<55),'CallLengthBin'] = 5
all_df['CallLengthBin'] = all_df['CallLengthBin'].astype(int)
############################################
all_df = all_df.drop('CallLenBand',axis=1)
############################################
print(" # récupérer l'heure début d'appel")
# récupérer l'heure début d'appel
all_df['CallStartHour'] = all_df['CallStart'].dt.hour
print(all_df[['CallStart','CallEnd','CallLength','CallStartHour']].head())
############################################

# Obtenir le jour ouvrable du dernier contact en fonction du jour et du mois de l'appel, en supposant que l'année soit 2016
all_df['LastContactDate'] = all_df.apply(lambda x:datetime.datetime.strptime("%s %s %s" %(2016,x['LastContactMonth'],x['LastContactDay']),"%Y %b %d"),axis=1)
all_df['LastContactWkd'] = all_df['LastContactDate'].dt.weekday
all_df['LastContactWkd'].value_counts()
all_df['LastContactMon'] = all_df['LastContactDate'].dt.month
all_df = all_df.drop('LastContactMonth',axis=1)

#############################################
# Obtenir la semaine de dernier contact
all_df['LastContactWk'] = all_df['LastContactDate'].dt.week


#############################################
# Obtenir le numéro de la semaine
MonWk = all_df.groupby(['LastContactWk', 'LastContactMon'])['Education'].count().reset_index()
MonWk = MonWk.drop('Education', axis=1)
MonWk['LastContactWkNum'] = 0
for m in range(1, 13):
    k = 0
    for i, row in MonWk.iterrows():
        if row['LastContactMon'] == m:
            k = k + 1
            row['LastContactWkNum'] = k


def get_num_of_week(df):
    for i, row in MonWk.iterrows():
        if (df['LastContactWk'] == row['LastContactWk']) & (df['LastContactMon'] == row['LastContactMon']):
            return row['LastContactWkNum']


all_df['LastContactWkNum'] = all_df.apply(lambda x: get_num_of_week(x), axis=1)
#print(all_df[['LastContactWkNum', 'LastContactWk', 'LastContactMon']].head(10))

#############################################
print("Avant LE et ONH")
print("all_df.shape:", all_df.shape)
#############################################
print(" save CSV LE")
all_df.to_csv('Data_LabelEncoding.csv')
#############################################
# Séparer les features numeriques et categoricques
cat_feats = all_df.select_dtypes(include=['object']).columns
num_feats = all_df.select_dtypes(include=['float64','int64']).columns
num_df = all_df[num_feats]
cat_df = all_df[cat_feats]
print('il y %d features numeriques et %d features categoriques\n' %(len(num_feats),len(cat_feats)))
print('Features numeriques:\n',num_feats.values)
print('Features categoriques:\n',cat_feats.values)

#############################################
# One hot encoding des attributs catégoriels
cat_df = pd.get_dummies(cat_df)

#############################################
# Merge tous les features
all_df = pd.concat([num_df,cat_df],axis=1)
#############################################
print("CSV apres ONH: ", all_df.shape)
all_df.to_csv('Data_OneHotEncoding.csv')
#############################################
idx=pd.IndexSlice
train_df=all_df.loc[idx[['train',],:]]
test_df=all_df.loc[idx[['test',],:]]
train_df = train_df.drop(train_df.index[1742])
test_df.to_csv(path+'train_OneHotEncoding.csv')
test_df.to_csv(path+'test_OneHotEncoding.csv')


train_label=train['CarInsurance']
train_label = train_label.drop(train_label.index[1742])
test_df.to_csv(path+'train_label_OneHotEncoding.csv')

print(train_df.shape)
print(len(train_label))
print(test_df.shape)
#############################################

#############################################

#############################################

#############################################

#############################################

#############################################

#############################################

#############################################

#############################################

