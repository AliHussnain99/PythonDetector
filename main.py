#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#from sklearn import datasets
from sklearn.model_selection import cross_val_score # for cross validation purpose
from sklearn.model_selection import cross_validate # for cross validation and multiple evaluation
from sklearn.metrics import confusion_matrix
#%reload_ext memory_profiler


# # Data Loading and Exploration



df =  pd.read_csv(r'C:\Users\hp\Desktop\fraudTest.csv')

df
df.shape

df = df.drop_duplicates()
df = df.dropna()

df.shape

df.info()


# # Data Visualization


categorical_columns = ['merchant', 'category', 'gender','city']

for column in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=column, palette='Set3')
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)
    plt.show()

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['dob'] = pd.to_datetime(df['dob'])

# Time-based analysis
plt.figure(figsize=(12, 5))
df['trans_date_trans_time'].dt.hour.plot(kind='hist', bins=24, rwidth=0.9, color='skyblue')
plt.title('Hourly Transaction Distribution')
plt.xlabel('Hour of the Day')
plt.ylabel('Frequency')
plt.show()


# Visualize the class distribution
plt.figure(figsize=(6, 4))
df['is_fraud'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Class Distribution (0: Non-Fraud, 1: Fraud)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Geospatial data - Scatter plot of transactions
plt.figure(figsize=(10, 8))
plt.scatter(df['merch_long'], df['merch_lat'], c=df['is_fraud'], cmap='coolwarm', alpha=0.5)
plt.title('Geospatial Distribution of Transactions (Fraud vs. Non-Fraud)')
plt.xlabel('Merchant Longitude')
plt.ylabel('Merchant Latitude')
plt.colorbar(label='0: Non-Fraud, 1: Fraud')
plt.show()


# # Data Featuring and Transformation

# Feature 1: Transaction Amount Decimal Part
df['amt_decimal'] = df['amt'] % 1

# Ensure 'trans_date_trans_time' is treated as a string
df['trans_date_trans_time'] = df['trans_date_trans_time'].astype(str)

# Feature 2: Age of Cardholder at the Time of Transaction
df['transaction_date'] = pd.to_datetime(df['trans_date_trans_time'].str.split(' ').str[0])
df['cardholder_age'] = (df['transaction_date'] - pd.to_datetime(df['dob'])).dt.days // 365

# Feature 3: Transaction Amount to City Population Ratio
df['amt_to_city_pop_ratio'] = df['amt'] / df['city_pop']

# Display the updated dataset with new features
print(df[['amt_decimal', 'cardholder_age', 'amt_to_city_pop_ratio']].head())


columns_to_drop = [
    'Unnamed: 0',        # An index or identifier
    'cc_num',            # Masked credit card numbers
    'trans_date_trans_time',  #We have unix_time
    'transaction_date',  # Same as unix_time
    'first',             # First name
    'last',              # Last name
    'street',            # Street address
    'city',              # City (state information is more relevant)
    'state',             # State (zip code and lat/long provide location info)
    'zip',               # Zip code (redundant with lat/long)
    'dob',               # Date of birth (we've calculated cardholder_age)
    'trans_num',         # Transaction number or identifier
]

# Drop the specified columns
df = df.drop(columns=columns_to_drop)

# Display the updated dataset
print(df.head())

testing_data=pd.read_csv (r'C:\Users\hp\Desktop\fraudTest.csv')

# Handle missing values (if any)
testing_data = testing_data.dropna()


# In[14]:


# Feature 1: Transaction Amount Decimal Part
testing_data['amt_decimal'] = testing_data['amt'] % 1

# Ensure 'trans_date_trans_time' is treated as a string
testing_data['trans_date_trans_time'] = testing_data['trans_date_trans_time'].astype(str)

# Feature 2: Age of Cardholder at the Time of Transaction
testing_data['transaction_date'] = pd.to_datetime(testing_data['trans_date_trans_time'].str.split(' ').str[0])
testing_data['cardholder_age'] = (testing_data['transaction_date'] - pd.to_datetime(testing_data['dob'])).dt.days // 365

# Feature 3: Transaction Amount to City Population Ratio
testing_data['amt_to_city_pop_ratio'] = testing_data['amt'] / testing_data['city_pop']

# Display the updated dataset with new features
print(testing_data[['amt_decimal', 'cardholder_age', 'amt_to_city_pop_ratio']].head())


testing_data = testing_data.drop(columns=columns_to_drop)


# Handle missing values (if any)
df = df.dropna()

# Encode categorical variables using LabelEncoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

categorical_columns = ['gender', 'merchant', 'category', 'job']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])
    testing_data[col] = label_encoder.fit_transform(testing_data[col])


# # Data Spliting



# Split the data into features (X) and the target variable (y)
X_train = df.drop(columns=['is_fraud'])
y_train = df['is_fraud']




X_val = testing_data.drop(columns=['is_fraud'])
y_val = testing_data['is_fraud']




independent_variables = df[['amt_decimal', 'cardholder_age', 'amt_to_city_pop_ratio']]
dependent_variable = df['is_fraud']




independent_variables




x_train, x_test, y_train, y_test = train_test_split(independent_variables, dependent_variable, test_size=0.2)




print('is_fraud shape is: ', df.shape)
print('x_train shape is: ',x_train.shape)
print('x_test shape is: ',x_test.shape)
print('y_train shape is: ',y_train.shape)
print('y_test shape is: ',y_test.shape)


# ## K-Folds



kf = KFold(n_splits = 5)
for train_index, test_index in kf.split(independent_variables, dependent_variable):
    x_train, x_test, y_train, y_test = independent_variables.iloc[train_index], independent_variables.iloc[test_index], dependent_variable.iloc[train_index], dependent_variable.iloc[test_index]
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)




from sklearn.model_selection import StratifiedKFold
stratified_folds = StratifiedKFold(n_splits = 5, shuffle=True)

for train_index, test_index in stratified_folds.split(independent_variables, dependent_variable):
    x_train, X_val, y_train, y_val = independent_variables.iloc[train_index], independent_variables.iloc[test_index], dependent_variable.iloc[train_index], dependent_variable.iloc[test_index]
    print(x_train.shape, X_val.shape, y_train.shape, y_val.shape)




accuracy_score_list = []

def classfication_evaluation(prediction, y_test):
    #%memit
    
    actual_predicted = pd.DataFrame(data= [prediction, y_test], index = ['predicted_value', 'actual_value']).transpose()
    cm = metrics.confusion_matrix(actual_predicted.actual_value, actual_predicted.predicted_value)
    accuracy_score = metrics.accuracy_score(actual_predicted.actual_value, actual_predicted.predicted_value)
    print('\n accuracy score \t\t', accuracy_score)
    accuracy_score_list.append(accuracy_score)
    print('\n confusion matrix', cm, sep = '\n\n')
    print('\n\n', metrics.classification_report(actual_predicted.actual_value, actual_predicted.predicted_value))
    
    tp, fn, fp, tn = confusion_matrix(actual_predicted.actual_value, actual_predicted.predicted_value).ravel()
    print('Number of true positives are :',tp)
    print('Number of false negatives are :',fn)
    print('Number of false positives are :',fp)
    print('Number of true negatives are :',tn)
    
    
# draw heatmap of confusion matrix
sns.heatmap(cm, annot=True, square=True, annot_kws = {'wrap': False ,'size': 15, 'rotation': 45},
            fmt='g',
            xticklabels=['Is Fraud', 'no fraud'],
            yticklabels=['Is Fraud', 'no fraud'],
            robust = True,
            cmap="YlGnBu",
            linewidths=5, linecolor='grey',
            cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix', fontdict={'fontsize': 15}, pad = 15)
    


# ## Logistic Regression

# In[25]:


reg = LogisticRegression()
reg = reg.fit(x_train, y_train)
prediction = reg.predict(x_test)
prediction


# In[26]:


reg.get_params()


# In[27]:


reg.predict_proba(x_test)


# In[30]:


classfication_evaluation(prediction, y_test)


# ## K nearest neighbour

# In[31]:


knn=neighbors.KNeighborsClassifier(n_neighbors=10, metric='minkowski')
fit=knn.fit(x_train, y_train)


# In[32]:


prediction=fit.predict(x_test)
prediction


# In[33]:


print('3NN accuracy:', metrics.accuracy_score(y_test, prediction))


# In[34]:


correct = prediction == y_test
correct.sum()


# In[35]:


actual_predicted = pd.DataFrame(data= [prediction, y_test, correct], index = ['predicted_value', 'actual_value', 'correct']).transpose()
print(actual_predicted.correct.value_counts())
actual_predicted.head(20)


# In[ ]:





# In[ ]:





# In[ ]:




