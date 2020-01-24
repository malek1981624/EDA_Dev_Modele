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

path='./data/'
df = pd.read_csv(path+'carInsurance_train.csv', encoding='utf8', sep=',')
df.set_index('Id')
print(  )
df.shape
print(  )
df.info()
print(  )

df['Age'].hist()
print(  )
df['Age'].value_counts(normalize=True).plot(kind='bar')
print(  )
print ("*"*50)
df['Marital'].value_counts().sort_index().plot.bar()

print(  )
print ("*"*50)
df['Education'].value_counts().sort_index().plot.bar()

print(  )

print(  )
df['Default'].value_counts().sort_index().plot.bar()

print(  )
df['CarLoan'].value_counts().sort_index().plot.bar()

print(  )
df['Communication'].value_counts().sort_index().plot.bar()

print(  )
df['LastContactMonth'].value_counts().sort_index().plot.bar()

print(  )
df['NoOfContacts'].value_counts().sort_index().plot.bar()

print(  )
df['Job'].value_counts().sort_index().plot.bar()

print(  )
df['PrevAttempts'].value_counts().sort_index().plot.bar()

print(  )
df['Outcome'].value_counts().sort_index().plot.bar()

print(  )
df['Job'].value_counts().sort_index().plot.bar()

print(  )
df['CarInsurance'].value_counts().sort_index().plot.bar()

print(  )
df.groupby('Age')['CarInsurance'].sum().plot(kind='bar')
plt.plot(df.Age,df.CarInsurance)
plt.title('Age/CarInsurance')
plt.xlabel("Age")
plt.ylabel("CarInsurance")
plt.show()
plt.savefig("Age-CarInsurance.png")

print(  )
df.groupby('Default')['CarInsurance'].sum().plot(kind='bar')
plt.plot(df.Default,df.CarInsurance)
plt.title('Default/CarInsurance')
plt.show()
print(  )
df.groupby('HHInsurance')['CarInsurance'].sum().plot(kind='bar')

print(  )
df.groupby('CarLoan')['CarInsurance'].sum().plot(kind='bar')

print(  )
df.groupby('Communication')['CarInsurance'].sum().plot(kind='bar')

print(  )
plt.figure(figsize=(10,7), dpi= 100)
sns.distplot(df.loc[df['Education'] == 'tertiary',"Age"], color="blue", label="tertiary", hist_kws={'alpha':.7}, kde_kws={'linewidth':3})
sns.distplot(df.loc[df['Education'] == 'primary',"Age"], color="red", label="primary", hist_kws={'alpha':.7}, kde_kws={'linewidth':3})
sns.distplot(df.loc[df['Education'] == 'secondary', "Age"], color="green", label="secondary", hist_kws={'alpha':.7}, kde_kws={'linewidth':3})
plt.ylim(0, 0.10)

# Decoration
plt.title("Plot Densité d'Age par Niveau d'Education", fontsize=22)
plt.legend()

plt.savefig("Age par Niveau d'Education.png")
plt.show()

print()

#résultats des campagnes précédentes
plt.figure(figsize=(10,7), dpi= 100)
sns.countplot(x="Outcome",hue='CarInsurance',data=df);
plt.title("résultats des campagnes précédentes", fontsize=22)
plt.legend()
plt.savefig("résultats des campagnes précédentes.png")
plt.show()

#################################################3
#résultats des campagnes précédentes selon l niveau d'éducation
plt.figure(figsize=(10,7), dpi= 100)
sns.countplot(x="Outcome",hue='Education',data=df);
plt.title("résultats des campagnes précédentes selon le niveau d'éducation", fontsize=22)
plt.legend()
plt.savefig("résultats des campagnes précédentes selon le niveau d'éducation.png")
plt.show()

###############################################
plt.figure(figsize=(10,7), dpi= 100)
df.hist(bins=50, figsize=(20,15))
plt.title("dispersion des numériques", fontsize=22)
plt.savefig("dispersion des numériques.png")
plt.show()

###################################################
sns.set(style="white")
plt.figure(figsize=(10,7), dpi= 100)
corr = df.corr()
plt.title("dispersion des numériques", fontsize=22)
plt.legend()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr,annot=True, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});

plt.savefig("dispersion des numériques.png")
plt.show()

###############################################
plt.figure(figsize=(10,8), dpi= 80)
sns.pairplot(df, kind="scatter", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
plt.savefig("Pearson_Correlation.png")
plt.show()

#################################################

# features importants
imp_feats = ['CarInsurance','Age','Balance','HHInsurance', 'CarLoan','NoOfContacts','DaysPassed','PrevAttempts']
sns.pairplot(df[imp_feats],hue='CarInsurance',size=2.5)
plt.savefig("features_importants.png")

plt.show()







