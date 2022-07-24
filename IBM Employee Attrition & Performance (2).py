#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, train_test_split
from sklearn.metrics import precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
import xgboost as xgb
import warnings
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# # Read data

# In[28]:


data = pd.read_csv("C:\\Users\\Sneha Gaikwad\\Desktop\\Employee-Attrition.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.columns    #'StandardHours', 'Over18', 'EmployeeCount'


# In[6]:


data.info()


# ### missing values visualization

# In[7]:


null_feat = pd.DataFrame(len(data['Attrition']) - data.isnull().sum(), columns = ['Count'])

trace = go.Bar(x = null_feat.index, y = null_feat['Count'] ,opacity = 0.8, marker=dict(color = 'lightgrey',
        line=dict(color='#000000',width=1.5)))

layout = dict(title =  "Missing Values")
                    
fig = dict(data = [trace], layout=layout)
py.iplot(fig)


# In[8]:


data.describe()


# In[9]:


data.corr()


# In[10]:


plot = sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
plt.rcParams["figure.figsize"] = [30, 30] 


# ### Reassign target and drop useless features

# In[11]:


# Reassign target
data.Attrition.replace(to_replace = dict(Yes = 1, No = 0), inplace = True)
# Drop useless feat
data= data.drop(['Over18', 'EmployeeCount', 'StandardHours'], axis=1)


# In[12]:


data.head()


# In[13]:


data.shape


# In[14]:


data.info()


# In[ ]:





# # Exploratory Data Analysis using automated EDA library

# In[15]:


import dtale


# In[16]:


#dtale.show(data)


# # Feature engineering and selection
# 

# In[17]:


#dtypes: int64(26), object(9)
data.columns


# In[18]:


#No. of unique values in each column
data.nunique()


# In[19]:


#Dropping some more variables with the help of correlation matrix


cor_matrix = data.corr().abs()
print(cor_matrix)


# In[20]:


upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
print(upper_tri)


# In[21]:


to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.75)]
print(to_drop)


# In[22]:


data = data.drop(['MonthlyIncome', 'PerformanceRating', 'TotalWorkingYears', 'YearsInCurrentRole', 'YearsWithCurrManager'], axis=1)
data.head()


# In[23]:


data.info()


# ### Correlation matrix

# In[24]:


#correlation
# correlation = data.corr()
# #tick labels
# matrix_cols = correlation.columns.tolist()
# #convert to array
# corr_array  = np.array(correlation)

# #Plotting
# trace = go.Heatmap(z = corr_array,
#                    x = matrix_cols,
#                    y = matrix_cols,
#                    colorscale='Viridis',
#                    colorbar   = dict() ,
#                   )
# layout = go.Layout(dict(title = 'Correlation Matrix for variables',
#                         autosize = False,
#                         #height  = 1400,
#                         #width   = 1600,
#                         margin  = dict(r = 0 ,l = 210,
#                                        t = 25,b = 210,
#                                      ),
#                         yaxis   = dict(tickfont = dict(size = 9)),
#                         xaxis   = dict(tickfont = dict(size = 9)),
#                        )
#                   )
# fig = go.Figure(data = [trace],layout = layout)
# py.iplot(fig)


# ### Remove collinear features

# In[25]:


# # Threshold for removing correlated variables
# threshold = 0.8

# # Absolute value correlation matrix
# corr_matrix = data.corr().abs()
# corr_matrix.head()

# # Upper triangle of correlations
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# upper.head()

# # Select columns with correlations above threshold
# to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

# print('There are %d columns to remove :' % (len(to_drop)))
# data = data.drop(columns = to_drop)

# to_drop


# #  Employee Attrition Prediction and Model score analysis

# In[26]:


#Importing required packages and libraries

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,auc,classification_report,roc_auc_score,plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, Lasso

from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from urllib.request import urlopen


# In[29]:


#once again dropping unecessary columns and columns with more than 0.75 correlaion
df = data.drop(['EmployeeCount','StandardHours','Over18','EmployeeNumber','PerformanceRating','TotalWorkingYears','YearsInCurrentRole','YearsWithCurrManager'], axis=1)


# In[30]:


#Assigning category values to numerical colomns
df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})
df.Education.replace({1: 'High School',2:'Undergrad',3:'Graduate',4:'Post Graduate',5:'Doctorate'},inplace=True)
cols = ["JobInvolvement", "JobSatisfaction"]
for col in cols:
    df[col].replace({1 : "Low",2 : "Medium",3 : "High",4 : "Very High"}, inplace = True)


# In[31]:


#features for attrition
var = []
for i in df.columns:
    var.append([i, df[i].nunique(), df[i].drop_duplicates().values])


# In[32]:


#Extracting categorical variables from the data
categorical = []
for col, value in df.iteritems():
    if value.dtype == 'object':
        categorical.append(col)
cat_var = df[categorical]


# In[33]:


#Converting categorical data to indicator variables
dummies = pd.get_dummies(cat_var)
dummies.head()


# In[34]:


dummies.shape


# In[35]:


#Normalizing the attrition variable x
x = MinMaxScaler().fit_transform(dummies)
y = df['Attrition'].values


# ## using Logistic Regression model from scratch

# In[36]:


#Split train and test data in 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[37]:


#Fitting model on training data
lr = LogisticRegression()
model_1 = lr.fit(X_train,y_train)
model_1


# In[38]:


#Storing model predictions on testing data
y_pred=lr.predict(X_test)


# In[42]:


#Using seaborn for ploting heatmap
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True,cmap ='coolwarm' )
plt.show()


# In[43]:


#Print various scores of model 1 i.e.logistic regression
accuracy_1 = accuracy_score(y_test,y_pred)
print("Accuracy of logistic regression model is :",accuracy_1)
print("Precision of logistic regression model is :",metrics.precision_score(y_test, y_pred))
print("Recall of logistic regression model is :",metrics.recall_score(y_test, y_pred))


# In[ ]:





# ## using K-Nearest Neighbours

# In[44]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42, stratify=y)


# In[45]:


n = np.arange(1,20)
train =np.empty(len(n))
test = np.empty(len(n))

for i,k in enumerate(n):
    model4 = KNeighborsClassifier(n_neighbors=k)
    model4.fit(X_train, y_train)
    train[i] = model4.score(X_train, y_train)
    test[i] = model4.score(X_test, y_test) 

plt.plot(n, test, label='Testing Accuracy',color = 'red')
plt.plot(n, train, label='Training accuracy',color = 'green')
plt.legend()
plt.xlabel("No. of neighbors")
plt.ylabel("Accuracy")
plt.show()


# In[46]:


model_2 = KNeighborsClassifier(n_neighbors=6)
model_2.fit(X_train,y_train)


# In[47]:


model_2.score(X_test,y_test)


# In[48]:


accuracy_2 = model_2.score(X_test,y_test)


# In[49]:


y_pred = model_2.predict(X_test)


# In[50]:


confusion_matrix(y_test,y_pred)


# In[51]:


pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[52]:


print(classification_report(y_test,y_pred))


# ## using Random Forest

# In[53]:


rf = RandomForestClassifier(random_state=1)


# In[54]:


model_3 = rf.fit(X_train, y_train)
y_pred_rf = model_3.predict(X_test)


# In[55]:


sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True,cmap ='coolwarm' )
plt.rcParams["figure.figsize"] = [5, 5]
plt.show()


# In[56]:


print(classification_report(y_test,y_pred_rf))


# In[57]:


plot_confusion_matrix(rf, X_test, y_test, normalize="true",cmap ='coolwarm')


# In[58]:


accuracy_3 = accuracy_score(y_test ,y_pred_rf)


# In[59]:


#Plotting relative importance of variables in determining attrition and primary area of focus 
imp = np.array(rf.feature_importances_)
cols = np.array(dummies.columns)
data={'variable_names':cols,'imp_variables':imp}
v = pd.DataFrame(data)
v.sort_values(by=['imp_variables'], ascending=False,inplace=True) 
plt.figure(figsize=(10,10))
sns.barplot(x=v['imp_variables'], y=v['variable_names'],color  = 'grey')
plt.title('Attrition variables comparison')
plt.xlabel('Variable score')
plt.ylabel('Attrition variables')


# ### model comparison

# In[60]:


compare = pd.DataFrame({'Model' : [  'Logistic Regression','K-Nearest Neighbours','Random Forest'],'Score' : [accuracy_1,accuracy_2,accuracy_3]})


# In[61]:


compare.sort_values(by = 'Score', ascending = False)


# In[62]:


features_imp(xgb_clf, 'features')


# In[ ]:




