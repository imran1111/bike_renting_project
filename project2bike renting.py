
# coding: utf-8

# In[1]:


# Importing Libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import chi2_contingency


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Setting Working Directory

os.chdir("D:\Data Science\edWisor\Bike-Renting-Analysis\data")


# In[4]:


# Loading Data

data = pd.read_csv('day.csv')


# # Exploratory data Analysis¶

# In[6]:


data.head()


# In[7]:


features = pd.DataFrame(data.columns)
#features.to_csv('features.csv')


# In[8]:


# Continous variables

cnames = ['temp','atemp','hum','windspeed']
cat_names = ['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']


# In[9]:


# Corelation between continous variables

corr = data[cnames].corr()
corr
#corr.to_csv('Correlations.csv')


# In[10]:


sns.distplot(data['temp'])
#plt.savefig('temp.png')


# In[11]:


sns.distplot(data['atemp'])
#plt.savefig('atemp.png')


# In[12]:


sns.distplot(data['hum'])
#plt.savefig('hum.png')


# In[13]:


sns.distplot(data['windspeed'])
#plt.savefig('windspeed.png')


# In[14]:


plt.figure(figsize=(24,16))
plt.scatter(data['instant'], data['cnt'])
plt.xlabel('Days from January,1,2011 to December,31,2012', fontsize = 20)
plt.ylabel('Count', fontsize =20)
#plt.savefig('RentCount.png')


# In[15]:


# Creating Dummy Variables for non-binary categorical variables

for i in ['season','mnth','weekday','weathersit']:
    temp = pd.get_dummies(data[i], prefix = i)
    data = data.join(temp)
    data.drop(i, axis =1,inplace = True)


# In[16]:


data.drop(['instant','dteday','season_1','mnth_1','weekday_0','weathersit_1'],axis=1,inplace = True)


# In[17]:


data.head()


# In[18]:


# Splitting the data into train and test sets

train,test = train_test_split(data,test_size =0.2, random_state =0)


# In[19]:


# Preparing Data for modelling

X_train = train.drop(['casual','registered','cnt','temp'],axis=1)
X_test = test.drop(['casual','registered','cnt','temp'],axis=1)
y_casual = train['casual']
y_registered = train['registered']
y_cnt = train['cnt']


# In[20]:


# Evaluation Functions

def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape
#Calculate MAPE

def RMSE(y_true, y_pred):
    rms = sqrt(mean_squared_error(y_true, y_pred))
    return rms
#Calculate RMSE


# # Regression Models

# # Multiple linear Regression

# In[21]:


from sklearn.linear_model import LinearRegression


# In[22]:


# Grid Search for best Parameters

reg_lm = LinearRegression()
params_lm = [{'copy_X':[True, False],
              'fit_intercept':[True,False],
              'normalize':[True, False]}]
grid_search_lm = GridSearchCV(reg_lm, param_grid = params_lm, cv =10, n_jobs =-1)
grid_search_lm = grid_search_lm.fit(X_train,y_cnt)


# In[23]:


grid_search_lm.best_score_


# In[24]:


grid_search_lm.best_params_


# In[25]:


# Training with best paramaeters

reg_lm_best = LinearRegression(copy_X=True, fit_intercept=True, normalize=True)
reg_lm_best.fit(X_train,y_cnt)


# In[26]:


# Evaluating on training set

y_pred_lm = reg_lm_best.predict(X_train)
mape1_lm = MAPE(y_cnt, y_pred_lm)
rmse1_lm = RMSE(y_cnt, y_pred_lm)
print('MAPE : {:.2f}'.format(mape1_lm))
print('RMSE : {:.2f}'.format(rmse1_lm))


# In[27]:


# Evaluating on Test Set

y_pred_lm = reg_lm_best.predict(X_test)
mape2_lm = MAPE(test['cnt'], y_pred_lm)
rmse2_lm = RMSE(test['cnt'], y_pred_lm)
print('MAPE : {:.2f}'.format(mape2_lm))
print('RMSE : {:.2f}'.format(rmse2_lm))


# # Supoort Vector Regressor

# In[28]:


from sklearn.svm import SVR


# In[29]:


# Grid Search for best Parameters

reg_svr = SVR()
param= [{'C':[1,10,100,1000],
         'kernel': ['rbf','linear'],
         'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]}]
grid_search_svr = GridSearchCV(reg_svr, param_grid = param, cv =10, n_jobs =-1)
grid_search_svr = grid_search_svr.fit(X_train,y_cnt)


# In[30]:


grid_search_svr.best_params_


# In[31]:


grid_search_svr.best_score_


# In[32]:


# Training with best paramaeters

reg_svr_best = SVR(C = 1000, kernel = 'linear', gamma = 0.0001)
reg_svr_best.fit(X_train,y_cnt)


# In[33]:


# Evaluating on training set

a = reg_svr_best.predict(X_train)
mape1_svr = MAPE(y_cnt,a)
rmse1_svr = RMSE(y_cnt,a)
print('MAPE : {:.2f}'.format(mape1_svr))
print('RMSE : {:.2f}'.format(rmse1_svr))


# In[34]:


# Evaluating on Test Set

y_pred = reg_svr_best.predict(X_test)
mape2_svr = MAPE(test['cnt'],y_pred)
rmse2_svr = RMSE(test['cnt'],y_pred)
print('MAPE : {:.2f}'.format(mape2_svr))
print('RMSE : {:.2f}'.format(rmse2_svr))


# # Decision Tree Regressor

# In[35]:


from sklearn.tree import DecisionTreeRegressor


# In[36]:


# Grid Search for best Parameters

reg_dt = DecisionTreeRegressor(random_state = 0)
params = [{'max_depth':[2,4,6,8,10,12,15],
           'max_features':['auto','sqrt'],
           'min_samples_leaf':[2,4,6,8,10]}]
grid_search_dt = GridSearchCV(reg_dt, param_grid = params, cv =10, n_jobs =-1)
grid_search_dt = grid_search_dt.fit(X_train,y_cnt)


# In[37]:


grid_search_dt.best_score_


# In[38]:


grid_search_dt.best_params_


# In[39]:


# Training with best parameters

reg_dt_best = DecisionTreeRegressor(random_state = 0, max_depth = 12,
                                    min_samples_leaf = 10, max_features = 'auto')
reg_dt_best.fit(X_train,y_cnt)


# In[40]:


# Evaluating on training set

b = reg_dt_best.predict(X_train)
mape1_dt = MAPE(y_cnt,b)
rmse1_dt = RMSE(y_cnt,b)
print('MAPE : {:.2f}'.format(mape1_dt))
print('RMSE : {:.2f}'.format(rmse1_dt))


# In[41]:


# Evaluating on test set

y_pred_dt = reg_dt_best.predict(X_test)
mape2_dt = MAPE(test['cnt'],y_pred_dt)
rmse2_dt = RMSE(test['cnt'],y_pred_dt)
print('MAPE : {:.2f}'.format(mape2_dt))
print('RMSE : {:.2f}'.format(rmse2_dt))


# # Random Forest Regressor

# In[42]:


from sklearn.ensemble import RandomForestRegressor


# In[43]:


# Grid Search for best Parameters

reg_rf = RandomForestRegressor(random_state = 0)
params_rf = [{'max_depth':[8,10,12,15],
              'max_features':['auto','sqrt'],
              'min_samples_leaf':[2,4,6,8,10],
              'n_estimators': [200, 500, 600],
              'oob_score':[True, False]}]
grid_search_rf = GridSearchCV(reg_rf, param_grid = params_rf, cv =10, n_jobs =-1)
grid_search_rf = grid_search_rf.fit(X_train,y_cnt)


# In[44]:


grid_search_rf.best_score_


# In[45]:


grid_search_rf.best_params_


# In[46]:


reg_rf_best = RandomForestRegressor(random_state = 0, max_depth = 15,
                                    max_features = 'auto', min_samples_leaf = 2,
                                    n_estimators = 600, oob_score = True)
reg_rf_best.fit(X_train,y_cnt)


# In[47]:


# Evaluating on training set

c = reg_rf_best.predict(X_train)
mape1_rf = MAPE(y_cnt,c)
rmse1_rf = RMSE(y_cnt,c)
print('MAPE : {:.2f}'.format(mape1_rf))
print('RMSE : {:.2f}'.format(rmse1_rf))


# In[48]:


# Evaluating on test set

y_pred_rf = reg_rf_best.predict(X_test)
mape2_rf = MAPE(test['cnt'],y_pred_rf)
rmse2_rf = RMSE(test['cnt'],y_pred_rf)
print('MAPE : {:.2f}'.format(mape2_rf))
print('RMSE : {:.2f}'.format(rmse2_rf))


# # Result

# In[49]:


result = pd.DataFrame()
result['Model'] = ['Multiple Linear Regressor','Support Vector Regressor',
                   'Decision Tree Regressor', 'Random Forest Regressor']
result['Training MAPE'] = [mape1_lm, mape1_svr, mape1_dt, mape1_rf]
result['Training RMSE'] = [rmse1_lm, rmse1_svr, rmse1_dt, rmse1_rf]
result['Test MAPE'] = [mape2_lm, mape2_svr, mape2_dt, mape2_rf]
result['Test RMSE'] = [rmse2_lm, rmse2_svr, rmse2_dt, rmse2_rf]
#result.to_csv('result.csv')


# In[50]:


result


# # Output using Selected Model i.e. Random Forest Regressor

# In[51]:


#pd.DataFrame(y_pred_rf).to_csv('Output.csv')

