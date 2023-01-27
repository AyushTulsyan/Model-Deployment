#!/usr/bin/env python
# coding: utf-8

# # Import Required Libraries

# In[44]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_selection import mutual_info_regression
import pickle


# # Dataset

# Airline dataset is imported.

# In[2]:


df = pd.read_csv("D:\Projects Datasets\Invistico_Airline.csv")
df


# # Reviewing of Dataset

# In[3]:


df.shape


# Dataset have:

# 129880 Rows

# 23 Columns

# In[4]:


df.info()


# Dataset have:

# 1 Float type column

# 17 Integer Column

# 5 Object column

# In[5]:


df.describe()


# Statistical insights of numerical columns are figure out including mean, mode , median, percentile, max & min.

# In[6]:


df.nunique().sort_values(ascending = False)


# Unique values of all columns are sorted in descending order.

# In[7]:


df.isnull().sum().sort_values(ascending = False)


# Total number of null values are arranged in descending order, arrival column have missing values 393.

# # Separating Feature and Target Variables

# In[8]:


target = pd.DataFrame(df['satisfaction'])


# In[9]:


target['satisfaction'] = target['satisfaction'].replace('satisfied','1')
target['satisfaction'] = target['satisfaction'].replace('dissatisfied','0')
target['satisfaction'] = target['satisfaction'].astype('int64')


# In[10]:


feature = df.drop('satisfaction', axis = 1)


# Target and Feature variables are separated from dataset. Target column contains binary categorical values which are converted into int type for further processing.

# # Separation of Num. & Cat. variable

# In[11]:


discrete_num = [col for col in feature.columns if feature[col].dtype !='O' and feature[col].nunique() <= 25]
print(discrete_num)
print('No. of discrete variable: ', len((discrete_num)))


# In[12]:


conti_num = [col for col in feature.columns if feature[col].dtype !='O' and feature[col].nunique() > 25]
print(conti_num)
print('No. of continuous variable: ', len(conti_num))


# In[13]:


cat_var = [col for col in feature.columns if feature[col].dtype =='O']
print(cat_var)
print('No. of categorical variable: ', len(cat_var))


# Feature variables are further divided into discrete, continuous and categorical variables for clear visualization and processing.

# # Visualization of Discrete Feature

# In[14]:


for i in discrete_num:
    sns.barplot(y = feature[i], x = target['satisfaction'])
    plt.show()


# Bar graph is plotted for comparison of customers satisfaction regarding discrete variables. Airline has done good job in almost domains to provide comfort to customers but still there are some areas in which improvements are required such as convenient in departure/arrival time, location of gates, food & drink.

# # Visualization of Continuous Feature

# In[15]:


for i in conti_num:
    sns.distplot(x = feature[i], kde = True)
    plt.xlabel(i)
    plt.show()


# Probability distributon curve is plot for the contionus variables, here we can observe that age group between 20-45 are more frequent user of airlines, on an average during a journey flight covers a distance of 2000km and approx dealy in arrival and departure of a flight is about 25min.

# # Visualization of Categorical Feature

# In[16]:


for i in cat_var:
    sns.barplot(x = feature[i], y = target['satisfaction'])
    plt.show()


# Bar graph is plotted for comparison between categorical and target variables, here we can see that with repect to gender females are more satisfied than males passenger, obviously loyal customers are more satisfied in compare to disloyal, persons who travel because of business issues are more likely to have satisfied than personal once, business class provide more comfort to the passenger rather than eco & eco plus.

# # Creating Pipelines

# In[17]:


discrete_num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy = 'most_frequent')),
    ('scalar', StandardScaler())
])


# In[18]:


conti_num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy = 'median')),
    ('scalar', StandardScaler())
])


# In[19]:


cat_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy = 'most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])


# Number of pipelines are created for discrete, continous and categorical variables to perform certain preprocessing task such as standard scaling of data, filling out null values with the help of simple imputer and converion of categorical variables to discrete type using one hot encoder.

# # Transformation

# In[20]:


transformer = ColumnTransformer(transformers=[
    ('tnf1', discrete_num_pipeline, discrete_num),
    ('tnf2', conti_num_pipeline, conti_num),
    ('tnf3', cat_pipeline, cat_var)
], remainder = 'passthrough')


# In[21]:


feature = transformer.fit_transform(feature)


# Transformation is used to pass discrete, continuous and categorical variables in their respective pipelines and then fit and transform the processed data in feature variable.

# # Splitting of Train & Test Data

# In[22]:


X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=25)


# Training and testing data is splitted in the ratio of 80:20 to train the model and ready for predictions.

# # Applying Classification Model

# In[23]:


pipeline_log= Pipeline([
    ('log_reg', LogisticRegression())
])


# In[24]:


pipeline_knn= Pipeline([
    ('knn_clsf', KNeighborsClassifier())
])


# In[25]:


pipeline_rf= Pipeline([
    ('rf_clsf', RandomForestClassifier())
])


# In[26]:


pipeline_gb= Pipeline([
    ('gb_clsf', GradientBoostingClassifier())
])


# In[27]:


pipeline_xgb= Pipeline([
    ('xgb_clsf', XGBClassifier())
])


# In[28]:


pipelines = [pipeline_log, pipeline_knn, pipeline_rf, pipeline_gb, pipeline_xgb]


# In[29]:


pipe_dict = {0: 'Logistic Regression', 1: 'KNN', 2: 'Random Forest', 3: 'Gradient Boosting', 4: 'XGB Classifier'}


# In[30]:


for pipe in pipelines:
    pipe.fit(X_train, y_train)


# In[31]:


for i,model in enumerate(pipelines):
    print("{} Train Accuracy: {}".format(pipe_dict[i], model.score(X_train, y_train)))


# In[32]:


for i,model in enumerate(pipelines):
    print("{} Test Accuracy: {}".format(pipe_dict[i], model.score(X_test, y_test)))


# Different classification models are used with the help of pipelines and the accuracy score of train and test data is computed based on different models.
# Model having tuned acuuracy score for both train and test is selected. Although here we can see that Random forest provide accurcay score of 1.0 in train data so that means here is some chances that the model is overfit.

# # Best model is deployed

# In[33]:


clsf = XGBClassifier()
clsf.fit(X_train, y_train)
# regr=LogisticRegression(random_state=2025).fit(X_train, y_train)


# Extreme Boost Classification algorithm is used to train the model on training data and predict values.

# In[34]:


train_pred = clsf.predict(X_train)
train_pred


# In[35]:


train_accuracy = accuracy_score(train_pred, y_train)
train_accuracy


# Predicted values over 80% training data is calculated and accuracy of model over training data is observed.

# In[36]:


test_prediction = clsf.predict(X_test)
test_prediction


# In[37]:


test_accuracy = accuracy_score(test_prediction, y_test)
test_accuracy


# Predicted values over 20% test data is calculated and accuracy of model over test data is observed.

# # Confusion Matrix

# In[38]:


matrix = confusion_matrix(test_prediction, y_test)
matrix


# In[39]:


plot = plot_confusion_matrix(clsf, X_test, y_test)


# Here confusion matrix is plotted to measure the performance of XGBClassifier by identifying the total number of correct and incorrect predictons, as true negative and positive, false negative and positive.

# # Classification Report

# In[40]:


report = classification_report(test_prediction, y_test)
print(report)


# Classification report is imported to measure performance of metric by observing precision, recall, f1-score and support values. Used to limit the values of FP, FN and both precision and recall.

# # ROC-AUC Curve

# In[41]:


y_pred_proba = clsf.predict_proba(X_test)[:,1]
y_pred_proba


# Probability of test data is predicted and threshold value is used to separated satisfied and dissatisfied customers. Generally 0.5 is the default threshold value.

# In[42]:


roc_auc_score(y_test, test_prediction)


# Roc-auc score is calculated between test data and predicted test data.

# In[43]:


fpr, tpr, threshold = metrics.roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="AUC Score="+str(auc))
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# Roc-auc curve is plotted between true positive rate and false positive rate.

# # Conclusion

# So far airlines are doing great in satisfying the needs of the passenger, in future to secure more of number of passengers, airlines have to work on some areas like reduce delay and waiting time in departure/arrival time, efficient location of gates and should provide good quality of food and drinks at affordable price.

# Airlines should focus more on the area of interest of age group between 20-45 as they are more frequent travellers apart from the its average flight distance is about 2000km so efficent use of fuel should be practised and should try to provide more comfort in personal travels and economic class in order to increase their satisfactory rate.

# To achieve this task XGBClassifier model is doing good job by predicting the customers segments in varying conditions.

# In[47]:


pickle.dump(clsf, open('model.pkl','wb'))


# In[48]:


model = pickle.load(open('model.pkl','rb'))


# In[ ]:




