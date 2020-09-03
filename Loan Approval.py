# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:19:58 2019

@author: Shantanu Bankoti
"""

import pandas as pd
import numpy as ny
data_df= pd.read_csv("C:\\Users\\Shantanu Bankoti\\Documents\\final project\\XYZCorp_LendingData.txt",sep='\t')
data=data_df[['annual_inc','delinq_2yrs',
              'dti','funded_amnt','grade','home_ownership',
              'installment','loan_amnt','pub_rec','revol_util','term','total_acc',
              'total_pymnt','total_rec_int','total_rec_prncp','total_rev_hi_lim',
              'acc_now_delinq','tot_coll_amt','tot_cur_bal','verification_status','emp_length','issue_d','default_ind']]
y=data['emp_length'].str.split()
data['emp_length']=y.str.get(0)
data['emp_length']=ny.where(data['emp_length']=='10+',11,data['emp_length'])
data['emp_length']=ny.where(data['emp_length']=='<',0.5,data['emp_length'])
data['emp_length']=data['emp_length'].astype(float)
data['emp_length'].mean()
data['emp_length'].fillna(7, inplace=True)
data['emp_length'].value_counts()
x=data['term'].str.split()
data['term']=x.str.get(0)
data['term']=data['term'].astype(int)
data['term']=data['term']/12

data['annual_inc'].isnull().sum()
data['delinq_2yrs'].isnull().sum()
data['dti'].isnull().sum()
data['funded_amnt'].isnull().sum()
data['grade'].isnull().sum()
data['home_ownership'].isnull().sum()
data['installment'].isnull().sum()
data['loan_amnt'].isnull().sum()
data['pub_rec'].isnull().sum()
data['term'].isnull().sum()
data['total_acc'].isnull().sum()
data['total_pymnt'].isnull().sum()
data['total_rec_int'].isnull().sum()
data['total_rec_prncp'].isnull().sum()
data['total_rev_hi_lim'].isnull().sum()
data['acc_now_delinq'].isnull().sum()
data['tot_coll_amt'].isnull().sum()
data['tot_cur_bal'].isnull().sum()
data['verification_status'].isnull().sum()
data['issue_d'].isnull().sum()

data['total_rev_hi_lim'].mean()
data['tot_coll_amt'].mean()
data['tot_cur_bal'].mean()

data['total_rev_hi_lim'].fillna(32163.57, inplace=True)
data['tot_coll_amt'].fillna(225.412, inplace=True)
data['tot_cur_bal'].fillna(139766.25, inplace=True)

import matplotlib
matplotlib.pyplot.boxplot(data['annual_inc'])
matplotlib.pyplot.boxplot(data['delinq_2yrs'])
matplotlib.pyplot.boxplot(data['dti'])
matplotlib.pyplot.boxplot(data['funded_amnt'])
matplotlib.pyplot.boxplot(data['grade'])
matplotlib.pyplot.boxplot(data['home_ownership'])
matplotlib.pyplot.boxplot(data['installment'])
matplotlib.pyplot.boxplot(data['loan_amnt'])
matplotlib.pyplot.boxplot(data['pub_rec'])
matplotlib.pyplot.boxplot(data['term'])
matplotlib.pyplot.boxplot(data['total_acc'])
matplotlib.pyplot.boxplot(data['total_pymnt'])
matplotlib.pyplot.boxplot(data['total_rec_int'])
matplotlib.pyplot.boxplot(data['total_rec_prncp'])
matplotlib.pyplot.boxplot(data['total_rev_hi_lim'])
matplotlib.pyplot.boxplot(data['acc_now_delinq'])
matplotlib.pyplot.boxplot(data['tot_coll_amt'])
matplotlib.pyplot.boxplot(data['tot_cur_bal'])
matplotlib.pyplot.boxplot(data['verification_status'])
matplotlib.pyplot.boxplot(data['emp_length'])
matplotlib.pyplot.boxplot(data['issue_d'])

q1=data['annual_inc'].quantile(.05)
q3=data['annual_inc'].quantile(.95)
iqr=q3-q1
import numpy as ny
data['annual_inc']=ny.where(data['annual_inc']>(q3+iqr*1.5),(q3+iqr*1.5),data['annual_inc'])

q1=data['delinq_2yrs'].quantile(.05)
q3=data['delinq_2yrs'].quantile(.95)
iqr=q3-q1
import numpy as ny
data['delinq_2yrs']=ny.where(data['delinq_2yrs']>(q3+iqr*1.5),(q3+iqr*1.5),data['delinq_2yrs'])

q1=data['dti'].quantile(.05)
q3=data['dti'].quantile(.95)
iqr=q3-q1
import numpy as ny
data['dti']=ny.where(data['dti']>(q3+iqr*1.5),(q3+iqr*1.5),data['dti'])

q1=data['installment'].quantile(.05)
q3=data['installment'].quantile(.95)
iqr=q3-q1
import numpy as ny
data['installment']=ny.where(data['installment']>(q3+iqr*1.5),(q3+iqr*1.5),data['installment'])

q1=data['pub_rec'].quantile(.05)
q3=data['pub_rec'].quantile(.95)
iqr=q3-q1
import numpy as ny
data['pub_rec']=ny.where(data['pub_rec']>(q3+iqr*1.5),(q3+iqr*1.5),data['pub_rec'])

q1=data['total_acc'].quantile(.05)
q3=data['total_acc'].quantile(.95)
iqr=q3-q1
import numpy as ny
data['total_acc']=ny.where(data['total_acc']>(q3+iqr*1.5),(q3+iqr*1.5),data['total_acc'])

q1=data['total_pymnt'].quantile(.05)
q3=data['total_pymnt'].quantile(.95)
iqr=q3-q1
import numpy as ny
data['total_pymnt']=ny.where(data['total_pymnt']>(q3+iqr*1.5),(q3+iqr*1.5),data['total_pymnt'])

q1=data['total_rec_int'].quantile(.05)
q3=data['total_rec_int'].quantile(.95)
iqr=q3-q1
import numpy as ny
data['total_rec_int']=ny.where(data['total_rec_int']>(q3+iqr*1.5),(q3+iqr*1.5),data['total_rec_int'])


q1=data['total_rec_prncp'].quantile(.05)
q3=data['total_rec_prncp'].quantile(.95)
iqr=q3-q1
import numpy as ny
data['total_rec_prncp']=ny.where(data['total_rec_prncp']>(q3+iqr*1.5),(q3+iqr*1.5),data['total_rec_prncp'])


q1=data['total_rev_hi_lim'].quantile(.05)
q3=data['total_rev_hi_lim'].quantile(.95)
iqr=q3-q1
import numpy as ny
data['total_rev_hi_lim']=ny.where(data['total_rev_hi_lim']>(q3+iqr*1.5),(q3+iqr*1.5),data['total_rev_hi_lim'])

q1=data['acc_now_delinq'].quantile(.05)
q3=data['acc_now_delinq'].quantile(.95)
iqr=q3-q1
import numpy as ny
data['acc_now_delinq']=ny.where(data['acc_now_delinq']>(q3+iqr*1.5),(q3+iqr*1.5),data['acc_now_delinq'])


q1=data['tot_coll_amt'].quantile(.05)
q3=data['tot_coll_amt'].quantile(.95)
iqr=q3-q1
import numpy as ny
data['tot_coll_amt']=ny.where(data['tot_coll_amt']>(q3+iqr*1.5),(q3+iqr*1.5),data['tot_coll_amt'])


q1=data['tot_cur_bal'].quantile(.05)
q3=data['tot_cur_bal'].quantile(.95)
iqr=q3-q1
import numpy as ny
data['tot_cur_bal']=ny.where(data['tot_cur_bal']>(q3+iqr*1.5),(q3+iqr*1.5),data['tot_cur_bal'])

z=data['issue_d'].str.split('-')
z.str.get(1)
z.str.get(0)
g=z.str.get(0)
g[:]=ny.where(g[:]=='Jan',.01,g[:])
g[:]=ny.where(g[:]=='Feb',.02,g[:])
g[:]=ny.where(g[:]=='Mar',.03,g[:])
g[:]=ny.where(g[:]=='Apr',.04,g[:])
g[:]=ny.where(g[:]=='May',.05,g[:])
g[:]=ny.where(g[:]=='Jun',.06,g[:])
g[:]=ny.where(g[:]=='Jul',.07,g[:])
g[:]=ny.where(g[:]=='Aug',.08,g[:])
g[:]=ny.where(g[:]=='Sep',.09,g[:])
g[:]=ny.where(g[:]=='Oct',.1,g[:])
g[:]=ny.where(g[:]=='Nov',.11,g[:])
g[:]=ny.where(g[:]=='Dec',.12,g[:])

d=z.str.get(1)
g=g.astype(float)
d=d.astype(float)
v=g+d
v
data['issue_d']=v
data['issue_d'].astype(float)
type(data['issue_d'])
df=data.sort_values(by='issue_d')
df=df.drop(columns=['issue_d'])

colname=df.columns[:]
colname
#label encoder and # one HOTencoding
from sklearn import preprocessing
le={}
for x in colname:
    le[x]=preprocessing.LabelEncoder()
df
for x in colname:
   df[x]=le[x].fit_transform(df.__getattr__(x))
df
df.head()
X= df.values[:,:-1]
Y= df.values[:,-1]
from sklearn.preprocessing import StandardScaler# min max and  z scaling
scaler=StandardScaler()
scaler.fit(X)
X= scaler.transform(X)
X

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train, Y_test= train_test_split(X,Y,test_size=0.3,random_state=10)

from sklearn.tree import DecisionTreeClassifier
model_DecisionTree= DecisionTreeClassifier()
model_DecisionTree.fit(X_train,Y_train)
Y_pred = model_DecisionTree.predict(X_test)
print(list(zip(Y_test,Y_pred)))
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

from sklearn import tree
with open("car_model_DecisionTree.txt","w") as G:
    G=tree.export_graphviz(model_DecisionTree,out_file=G)
    
#predicting using a bagging classifier
from sklearn.ensemble import ExtraTreesClassifier

model=(ExtraTreesClassifier(21))
#fit the model on the data and predict the values
model=model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)    
    
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

#predicting using a random forest classifier
from sklearn.ensemble import RandomForestClassifier

model_RandomForest=RandomForestClassifier(500)
#fit the model on the data and predict the values
model_1=model_RandomForest.fit(X_train,Y_train)
Y_pred=model_1.predict(X_test)    
    
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

#prediction using adaboost classification
from sklearn.ensemble import AdaBoostClassifier
model_AdaBoost=AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
#fit the model on the data and predict the values
model_AdaBoost=model_AdaBoost.fit(X_train,Y_train)
Y_pred=model_AdaBoost.predict(X_test)    
    
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

#predicting using gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
model_GradientBoosting=GradientBoostingClassifier()
#fit the model on the data and predict the values
model_GradientBoosting=model_GradientBoosting.fit(X_train,Y_train)
Y_pred=model_GradientBoosting.predict(X_test)    
    
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
#creating the sub models
estimators=[]
model1=LogisticRegression()
estimators.append(('log',model1))
model2=DecisionTreeClassifier()
estimators.append(('cart',model2))
model3=SVC()
estimators.append(('svm',model3))
#create the ensemble model
ensemble=VotingClassifier(estimators)
ensemble.fit(X_train,Y_train)
Y_pred=ensemble.predict(X_test)
#print(Y_pred)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))


# roc_auc_score
# ----Compute the area under the ROC curve
# average_precision_score
# ----Compute average precision from prediction scores
# precision_recall_curve
# ----Compute precision-recall pairs for different probability thresholds


from sklearn.metrics import auc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

# calculate AUC
auc = roc_auc_score(Y_test, Y_pred)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
pyplot.show()

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc


# predict class values
yhat = ensemble.predict(X_test)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(Y_test, Y_pred)
# calculate F1 score
f1 = f1_score(Y_test, yhat)
# calculate precision-recall AUC
# auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(Y_test, Y_pred)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
pyplot.plot(recall, precision, marker='.')
# show the plot
pyplot.show()