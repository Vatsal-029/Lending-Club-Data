"""
@author: Vatsal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

df = pd.read_table(r'C:\Users\Vatsal\Desktop\projects\Assignments\Python Project\XYZCorp.txt', sep='\t')
df.head()

df.shape

df_rev = pd.DataFrame(df)
df_rev.isnull().sum()

# Creating dataframe of variables of df_rev that have missing values more than 40%
na = df_rev.isnull().sum()
na = na[na.values>(0.4*len(df))]

# Variable names that have more than 40% null values
na.index

# Dropping the above and some other insignificant variables
df_rev.drop(['desc', 'mths_since_last_delinq', 'mths_since_last_record',
       'mths_since_last_major_derog','dti_joint',
       'verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m',
       'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util',
       'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi',
       'total_cu_tl', 'inq_last_12m','id','member_id','policy_code','emp_title',
       'title','zip_code','next_pymnt_d','addr_state','emp_length','pymnt_plan','acc_now_delinq'],
       axis=1,inplace=True)
df_rev.shape

# Dropping rows which has null values since both are date variables and it doesn't make sense imputing them
df_rev.dropna(subset=['last_pymnt_d'], how='all', inplace=True)
df_rev.dropna(subset=['last_credit_pull_d'], how='all', inplace=True)

# Converting some variables to date format as the problem statement specifies that we have to split the data on basis of issue_d
df_rev['issue_d'] = pd.to_datetime(df_rev['issue_d'])
df_rev.dtypes

df_rev.isnull().sum()

# Imputing missing values by median (because of only numeric variables missing)
for x in df_rev.columns[:]:
    if df_rev[x].dtype=='int64' or df_rev[x].dtype=='float64':
        df_rev[x].fillna(df_rev[x].median(), inplace=True)
    
df_rev.isnull().sum()

df_rev.describe()

# We will delete two rows because the min value of annual income mentioned is 0 which is not possible, and no bank will give loan to anyone with 0 income
df_rev[df_rev['annual_inc'] == 0.0]

# Row number 462577 and 508976 have nill annual income, so we delete those rows
df_rev.drop([462577,508976], axis=0, inplace=True)

df_rev = df_rev.rename_axis('Index_num').sort_values(by = ['issue_d', 'Index_num'], ascending = [True, True])

df_rev.reset_index(inplace = True) 

df_rev.drop(['Index_num'], axis=1, inplace=True)

print(df_rev.shape)
df_rev.head()

df_rev.tail()

# Finding row number that end's on 1-05-2015 as we are asked to split the data from:
# (01-06-2007 to 01-05-2015) <-- Train Test --> (01-06-2015 to 01-12-2015)
df_rev[df_rev['issue_d']=='2015-05-01']

# There are 30,891 rows starting from 567,588 till 598,478
# In this case we will split the data based on the index number:
## train (0 to 598478) and test (598479 to 847055)

# Now that we know the index number from which we will split the data, we can drop issue_d variable
df_rev.drop(['issue_d'], axis=1, inplace=True)
df_rev.shape

colname = []
for j in df_rev.columns[:]:
    if df_rev[j].dtype == 'object':
        colname.append(j)
print(colname)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for j in colname:
    df_rev[j] = le.fit_transform(df_rev[j].astype(str))



# Creating independent (X) and dependent (Y) variables in array, inorder to make processing time less
#X = df_rev.values[:,:-1]
#Y = df_rev.values[:,-1]
#Y = Y.astype(int)

X_train = df_rev.values[:598478,:-1]
X_test = df_rev.values[598478:,:-1]
Y_train = df_rev.values[:598478,-1]
Y_test = df_rev.values[598478:,-1]

print('X_train shape: ',X_train.shape)
print('X_test shape: ',X_test.shape)
print('Y_train shape: ',Y_train.shape)
print('Y_test shape: ',Y_test.shape)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)

### Model Building

# Logistic Regression

from sklearn.linear_model import LogisticRegression
#create a model
classifier = LogisticRegression()
#fitting training data to model
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
print(list(zip(Y_test, Y_pred)))
print()
print(classifier.coef_)
print()
print(classifier.intercept_)

# Classification Report - Accuracy and Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm = confusion_matrix(Y_test, Y_pred)
print(cfm)
print('Classification Report: ')
print(classification_report(Y_test, Y_pred))
acc = accuracy_score(Y_test, Y_pred)
print('Model Accuracy: ',acc)

# Adjusting the threshold
# store the predicted probabilities
y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)

#Setting threshold to 0.4
y_pred_class = []
for value in y_pred_prob[:,1]:
    if value > 0.40:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
        
print(y_pred_class)

# Checking accuracy and confusion matrix for threshold at 0.40
cfm = confusion_matrix(Y_test, y_pred_class)
print(cfm)
print('Classification Report: ')
print(classification_report(Y_test, y_pred_class))
acc = accuracy_score(Y_test, y_pred_class)
print('Threshold at 0.40 model accuracy: ', acc)

# Checking type1 and type2 error for different thresholds
for a in np.arange(0,1,0.01):
    predict_mine=np.where(y_pred_prob[:,1]>a,1,0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print('Errors at threshold',a,':',total_err,",Type 2 error:",cfm[1,0],',Type 1 error:',cfm[0,1])


# ROCR curve
from sklearn import metrics
fpr,tpr,z = metrics.roc_curve(Y_test, y_pred_class)
auc = metrics.auc(fpr,tpr)
print('Area Under the Curve(AUC): ',auc)
print('False Positive Rate(FPR): ',fpr)
print('True Positive Rate(TPR): ',tpr)

# ROC Curve
plt.title('Receiver Operating Characteristics (ROC)')
plt.plot(fpr,tpr,'b',label=auc)
plt.legend(loc='best')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# AUC with probability values
fpr,tpr,z = metrics.roc_curve(Y_test, y_pred_prob[:,1])
auc = metrics.auc(fpr,tpr)
print('Area Under the Curve(AUC): ',auc)
print(z)

plt.title('Receiver Operating Characteristics (ROC)')
plt.plot(fpr,tpr,'b',label=auc)
plt.legend(loc='best')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# Using Cross-Validation for Logistic Regression
classifier = (LogisticRegression())
#performing kfold_cross_validation
from sklearn.model_selection import KFold
kfold_cv = KFold(n_splits=10)
print(kfold_cv)

from sklearn.model_selection import cross_val_score
#running the model using scoring metric as accuracy
kfold_cv_result = cross_val_score(estimator=classifier,X=X_train,y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
# finding the mean of kfold_cv_result
print(kfold_cv_result.mean())

# We now try running the data with a new algorithm

# Decision Tree

X_train = df_rev.values[:598478,:-1]
X_test = df_rev.values[598478:,:-1]
Y_train = df_rev.values[:598478,-1]
Y_test = df_rev.values[598478:,-1]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)

# Predicting using decision_tree_classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=10, min_samples_leaf=5,
                           max_depth=10)
#fit the model on the data and predict the values
dt.fit(X_train, Y_train)
#predicting Y values
Y_pred = dt.predict(X_test)

# Classification report and Accuracy score check
cfm = confusion_matrix(Y_test, Y_pred)
print(cfm)
print('Classification Report: ')
print(classification_report(Y_test, Y_pred))
acc = accuracy_score(Y_test, Y_pred)
print('Decision Tree Accuracy Score: ', acc)

# Checking the important features according to the model
print(list(zip(colname,dt.feature_importances_)))

# using cross-validation on Decision tree model
classifier = (DecisionTreeClassifier())
#performing kfold_cross_validation
kfold_cv = KFold(n_splits=5)
print(kfold_cv)

from sklearn.model_selection import cross_val_score
kfold_cv_result = cross_val_score(estimator=classifier,X=X_train,
                                 y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
print(kfold_cv_result.mean())

for train_value, test_value in kfold_cv.split(X_train):
    classifier.fit(X_train[train_value],Y_train[train_value]).predict(X_train[test_value])
    
Y_pred = classifier.predict(X_test)
#print(list(zip(Y_test, Y_pred)))

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm = confusion_matrix(Y_test,Y_pred)
print(cfm)
print("Classification report : ")
print(classification_report(Y_test,Y_pred))
acc = accuracy_score(Y_test,Y_pred)
print("Cross Validation Decision Tree Accuracy : ",acc)

from sklearn import tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (14,14), dpi=300)
tree.plot_tree(classifier, filled=True);
fig.savefig('decisionTree_xyz_corp.png')