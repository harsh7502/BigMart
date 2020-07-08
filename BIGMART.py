# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:56:17 2020

@author: agarw
"""

'''https://www.analyticsvidhya.com/blog/2016/02/bigmart-sales-solution-top-20/'''
'''https://www.kaggle.com/littleraj30/big-mart-sale-prediction-in-depth-ensemble'''
'''Import Library'''
import pandas as pd
import numpy as np
import seaborn as sns

train=pd.read_csv("train_bigmart.csv")
test=pd.read_csv("test_bigmart.csv")

train['source']='train'
test['source']='test'
'''combining train and test'''
df=pd.concat([train,test],ignore_index=True)

df.isnull().sum()

df.dtypes

'''filling item_weight nan value'''
id_wt=df.groupby('Item_Identifier').mean()['Item_Weight'].to_frame()

#id_wt.loc['FDY38']

def nafiller(col):
    if(np.isnan(col[1])):
        print(col[0][0])
        return(id_wt.loc[col[0]][0])
    else:
        return(col[1])

df['Item_Weight']=df[['Item_Identifier','Item_Weight']].apply(nafiller,axis=1)


'''converting years to age'''


df['Outlet_Age']=2013-df['Outlet_Establishment_Year']
df.drop('Outlet_Establishment_Year',axis=1,inplace=True)
'''checking and filling nan dependence of outlet size'''
temp1=pd.crosstab(df['Outlet_Location_Type'],df['Outlet_Size'])
temp1.plot(kind='bar',stacked=False)


temp2=pd.crosstab(df['Outlet_Type'],df['Outlet_Size'])
temp2.plot(kind='bar',stacked=False)

def nafiller2(col):
    if(col[0]=='Supermarket Type2' or col[0]=='Supermarket Type3'):
        return('Medium')
    elif(col[0]=='Grocery Store' or col[2]=='Tier 2'):
        return('Small')
    else:
        return(col[1])
        

outlet_data=df[['Outlet_Type','Outlet_Size','Outlet_Location_Type']]

df['Outlet_Size']=outlet_data.apply(nafiller2,axis=1)


'''Some fat content were wrongly named correcting it'''

df['Item_Fat_Content'].unique()

df['Item_Fat_Content']=df['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat', 'Low Fat','Regular'])


'''Visibility of some product was 0 analysizing and correcting'''


df['Item_Visibility'].value_counts()

df.loc[df['Item_Visibility']==0]

df['Item_Identifier'].value_counts()
#df.loc[df['Item_Identifier']=='NCD19']['Item_Visibility']
sns.boxplot(df['Outlet_Size'],df['Item_Visibility'])

sns.lineplot(df['Item_MRP'],df['Item_Visibility'])


def zeroreplacer(col):
    if(col[1]==0):
        if(col[0]=='Grocery Store'):
            return(0.08)
        else:
            return(0.065)
    return(col[1])
        

df['Item_Visibility']=df[['Outlet_Type','Item_Visibility']].apply(zeroreplacer,axis=1)



''' Categorizing based on item id provided'''

#train.groupby(['Outlet_Identifier'])['Outlet_Type'].agg(pd.Series.mode

#Get the first two characters of ID:
df['Item_Type_Combined'] = df['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
df['Item_Type_Combined'] = df['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
df['Item_Type_Combined'].value_counts()

sns.boxplot(df['Item_Type'],df['Item_MRP'])



'''there was non consumable items correcting there item fat content as non edible'''



df.loc[df['Item_Type_Combined']=='Non-Consumable','Item_Fat_Content']='Non_Edible'


'''broadly categorizing item based on mrp''' 

df['Item_Fat_Content'].value_counts()
df['Item_Identifier'].value_counts()
df['outlet']=df['Outlet_Identifier']
df['Item_Type'].unique()


itm_type_mrp=df.groupby(['Item_Type'])['Item_MRP'].mean().to_frame()
itm_type_mrp.loc['Canned']
df['Item_Grp']=df['Item_Type'].apply(lambda x:1 if itm_type_mrp.loc[x][0]>140 else 0)
df['Item_Grp']=df['Item_Grp'].astype('object')
df.drop(['Item_Type'],axis=1,inplace=True)


'''Lets try'''
df.corr()['Item_Outlet_Sales']
sns.heatmap(df.corr())
'''drop because correlation very poor'''
df.drop(['Item_Weight'],axis=1,inplace=True)



'''remove outlier'''
sns.boxplot(df['Item_Fat_Content'],df['Item_Outlet_Sales'])
df.loc[df['Item_Fat_Content']=='Non_Edible','Item_Outlet_Sales'].max()
ind=df.loc[df['Item_Outlet_Sales']>10000].index
df.drop(index=ind,inplace=True)



sns.boxplot(df['Outlet_Size'],df['Item_Outlet_Sales'])

df.loc[df['Outlet_Size']=='High','Item_Outlet_Sales'].max()
ind=df.loc[df['Item_Outlet_Sales']==9069.5276].index
df.drop(index=ind,inplace=True)


sns.boxplot(df['Outlet_Location_Type'],df['Item_Outlet_Sales'])
df.loc[df['Outlet_Location_Type']=='Tier 1','Item_Outlet_Sales'].max()
ind=df.loc[df['Item_Outlet_Sales']==9779.9362].index
df.drop(index=ind,inplace=True)


sns.boxplot(df['Item_Type_Combined'],df['Item_Outlet_Sales'])
df.loc[df['Item_Type_Combined']=='Drinks','Item_Outlet_Sales'].max()
ind=df.loc[df['Item_Outlet_Sales']==9554.23].index
df.drop(index=ind,inplace=True)

#sns.distplot(df['Item_Outlet_Sales'])
#sns.boxplot(df['Item_Type'],df['Item_Outlet_Sales'])

sns.scatterplot(df['Item_Visibility'],df['Item_Outlet_Sales'])
sns.scatterplot(df['Outlet_Age'],df['Item_Outlet_Sales'])

'''converting logarthmic to gaussian'''
sns.distplot(np.log(df['Item_Visibility']))
df['Item_Visibility']=np.log(df['Item_Visibility'])


'''we observe bins of mrp so we devide it in bins'''
sns.scatterplot(df['Item_MRP'],df['Item_Outlet_Sales'],hue=df['Item_Type_Combined'])
df['Item_MRP_Grouped']=pd.cut(df.Item_MRP,bins=[25,69,137,203,270],labels=['a','b','c','d'],right=True)
df.drop(['Item_MRP'],axis=1,inplace=True)
df['Item_MRP_Grouped']=df['Item_MRP_Grouped'].astype('object')
df['Item_MRP_Grouped'].dtypes

df1=df.copy()

sns.boxplot(df['outlet'],df['Item_Outlet_Sales'])


'''categoricalvariablehandling'''



categorical_columns = list(df.columns[df.dtypes == 'object'])
usefull = ['Item_Identifier','Outlet_Identifier','source']
for i in usefull:
    categorical_columns.remove(i)
#from sklearn import preprocessing 
#label_encoder = preprocessing.LabelEncoder() 
df['Outlet_Location_Type']=label_encoder.fit_transform(df['Outlet_Location_Type'])

df['Outlet_Type']=label_encoder.fit_transform(df['Outlet_Type'])

df['Item_MRP_Grouped']=label_encoder.fit_transform(df['Item_MRP_Grouped'])

df['outlet'].unique()
  
#for column in categorical_columns:
#    df[column]= label_encoder.fit_transform(df[column])    


df['Item_Fat_Content']=df['Item_Fat_Content'].replace(['Regular','Low Fat','Non_Edible'],[2,1,0]).astype('object')
df['Outlet_Size']=df['Outlet_Size'].replace(['High','Medium','Small'],[2,1,0]).astype('object')

df=df.join(pd.get_dummies(df['Item_Type_Combined'],drop_first=True))
df.drop('Item_Type_Combined',axis=1,inplace=True)
df2=df.copy()
#df=df.join(pd.get_dummies(df['outlet'],drop_first=True))
df.drop('outlet',axis=1,inplace=True)

out_sale=df.groupby('Outlet_Identifier').mean()['Item_Outlet_Sales']
sns.boxplot(df['Outlet_Identifier'],df['Item_Outlet_Sales'])
df['Outlet_combined']=df['Outlet_Identifier'].apply(lambda x:0 if out_sale.loc[x]>3000 else(2 if out_sale.loc[x]<1994 else 1))
df['Outlet_combined']=df['Outlet_combined'].astype('object')

'''Data Seperation back into train and test'''


train_modified=df.loc[df['source']=='train']
test_modified=df.loc[df['source']=='test']
test_modified.drop(['source','Item_Outlet_Sales'],axis=1,inplace=True)
train_modified.drop('source',axis=1,inplace=True)
train_copy=train_modified.copy()
test_copy=test_modified.copy()

'''Let's Start ml'''

X_train=train_modified.drop(['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'],axis=1)
y_train=train_modified['Item_Outlet_Sales']

X_test=test_modified.drop(['Item_Identifier','Outlet_Identifier'],axis=1)

'''Scaling jb linear krna wrna nahi'''
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc1=StandardScaler()
Scaled_x_train=sc.fit_transform(X_train)
scaled_x_test=sc1.fit_transform(X_test)

from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor()
parametersk=[{'n_neighbors':[5,8,10,12,14,15,17],'weights':['uniform','distance'],'algorithm':['auto','ball_tree','kd_tree','brute']}]
from sklearn.model_selection import GridSearchCV
gscv=GridSearchCV(knn,parametersk,cv=5)
gscv.fit(Scaled_x_train,y_train)
gscv.best_estimator_
scores=gscv.cv_results_
scores['mean_test_score'].mean()

'''knn'''
knn1=KNeighborsRegressor(n_neighbors=14,metric='minkowski',p=2)
knn1.fit(Scaled_x_train,y_train)
y2=knn1.predict(scaled_x_test)

ss1=test_modified[['Item_Identifier','Outlet_Identifier']]
ss1['Item_Outlet_Sales']=y2
ss1.to_csv("ss1.csv")

'''randomforest'''
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=5, max_features='auto', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=50,
                      min_samples_split=15, min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=4, oob_score=False,
                      random_state=None, verbose=0, warm_start=False)
rf.fit(X_train,y_train)
y3=rf.predict(X_test)
ss1['Item_Outlet_Sales']=y3
ss1.to_csv("ss1.csv")

paramrf=[{'n_estimators':[70,85,100,150],'max_depth':[3,6,5,7],'min_samples_leaf':[30,50,80],'min_samples_split':[15,10,8]}]
gscv2=GridSearchCV(rf,paramrf,cv=5,scoring='neg_root_mean_squared_error')
gscv2.fit(X_train,y_train)
gscv2.best_estimator_
scores1=gscv2.cv_results_
scores1['mean_test_score'].mean()



rf=RandomForestRegressor()
rf.fit(X_train,y_train)
y3=rf.predict(X_test)
ss1['Item_Outlet_Sales']=y3
ss1.to_csv("ss1.csv")
score=cross_val_score(rf,train,train_label,cv=10,scoring='neg_mean_squared_error')
rf_score_cross=np.sqrt(-score)
np.mean(rf_score_cross),np.std(rf_score_cross)



'''linear regression'''

from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize=True)  
regressor.fit(X_train,y_train) #training the algorithm
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)
y_pred = regressor.predict(X_test)
ss1=test_modified[['Item_Identifier','Outlet_Identifier']]
sample = test[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']]
sample.to_csv('LinearRegressionSubmission.csv')

from sklearn.ensemble import xgboost
xgb=xgboost()

 