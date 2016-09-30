# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 12:33:10 2016

@author: s6324900
"""

#!/usr/bin/python

import pickle
import numpy as np
import copy

# Change path to Tools directory
import os
os.chdir(r"C:\Users\s6324900\Documents\ud120projects\tools")

from feature_format import featureFormat, targetFeatureSplit

# Change path to Final Project directory
os.chdir(r"C:\Users\s6324900\Documents\ud120projects\final_project")
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'total_payments', 'total_stock_value','restricted_stock',
                 'exercised_stock_options','restricted_stock_deferred', 'expenses', 'loan_advances', 'long_term_incentive', 
                 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi' ]
                
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Data Cleaning, Feature Engineering and Feature Transformation 
# Remove bogus keys - TOTAL and THE TRAVEL AGENCY IN THE PARK
# Removed key Lockhart Eugene as all values were zero
# First create a copy of the dictionary
data_dict2 = copy.deepcopy(data_dict)   
keys = sorted(data_dict2.keys())
print(keys)
print(len(keys))

if data_dict2.has_key('TOTAL'):
    del data_dict2['TOTAL']
    
if data_dict2.has_key('THE TRAVEL AGENCY IN THE PARK'):
    del data_dict2['THE TRAVEL AGENCY IN THE PARK']
    
if data_dict2.has_key('LOCKHART EUGENE E'):
    del data_dict2['LOCKHART EUGENE E']
    
keys = sorted(data_dict2.keys())
print(len(keys))
# 143 keys after removing above 3 keys 

#List of features
data_dict.get('LOCKHART EUGENE E')
data_dict2.get('MCDONALD REBECCA')

# Replace nan with zero for all the select features
def remove_nan(dictionary,features):
    keys = sorted(dictionary.keys())
    for feature in features:
        n=0
        tmp_list1 = []
        tmp_list2 = []
        for key in keys:
            value = dictionary[key][feature]
            tmp_list1.append( float(value) )
        
        #convert list to numpy array
        arr = np.array(tmp_list1)
        # Replace nan with zero
        arr[np.isnan(arr)] = 0
        #Convert array back to list 
        tmp_list2 = np.array(arr).tolist()
        #print(tmp_list2)
        
        # Assign back to same dictionary or a new one        
        for key in keys:
            dictionary[key][feature] = tmp_list2[n]
            n +=1
        
    #return dictionary
#feature_list2 = ['salary']  
remove_nan(data_dict2, features_list)
#Check that above has happened. NAN will only be removed for our selected features_list
data_dict2.get('MCDONALD REBECCA')

# Add three new features
# 1. perc_emails_to_poi: 'from_this_person_to_poi'/ 'to_messages'
# 2: perc_emails_from_poi : 'from_poi_to_this_person'/'from_messages'
# 3: perc_emails_shared_poi: 'shared_receipt_with_poi'/'from_messages'

def add_features(dictionary):
    keys = sorted(dictionary.keys())
    for key in keys:
        if dictionary[key]['to_messages'] > 0 :
            dictionary[key]['perc_emails_to_poi'] = float(dictionary[key]['from_this_person_to_poi'])/ float(dictionary[key]['to_messages'])
        else:
            dictionary[key]['perc_emails_to_poi'] = 0
        if dictionary[key]['from_messages'] > 0:
            dictionary[key]['perc_emails_from_poi'] = float(dictionary[key]['from_poi_to_this_person'])/ float(dictionary[key]['from_messages'])
            dictionary[key]['perc_emails_shared_poi'] = float(dictionary[key]['shared_receipt_with_poi'])/ float(dictionary[key]['from_messages'])
        else:
            dictionary[key]['perc_emails_from_poi'] = 0
            dictionary[key]['perc_emails_shared_poi'] = 0
    #return dictionary

add_features(data_dict2)
# Check new features are added
data_dict2.get('MCDONALD REBECCA')

# New feature_list with the three new features added. Removed other email features
features_list_new = ['poi','salary', 'bonus', 'total_payments', 'total_stock_value','restricted_stock',
                 'exercised_stock_options','restricted_stock_deferred', 'expenses', 'loan_advances', 'long_term_incentive', 
                   'perc_emails_to_poi','perc_emails_from_poi','perc_emails_shared_poi']

# Clean outliers - floor and cap outliers 
def outlier_removal(dictionary, features):
    fac = 1
    keys = sorted(dictionary.keys())
    for feature in features:
        n=0
        tmp_list1 = []
        tmp_list2 = []
        for key in keys:
            value = dictionary[key][feature]
            tmp_list1.append( float(value) )
        
        #convert list to numpy array
        arr = np.array(tmp_list1)
        # Replace nan with zero
        arr[np.isnan(arr)] = 0
        m = np.mean(arr)
        s = np.std(arr)
        #Outlier removal - floor and cap values
        arr[arr > m +fac*s] = m +fac*s
        arr[arr< m - fac*s] = m- fac*s
        #Convert array back to list 
        tmp_list2 = np.array(arr).tolist()
        #print(tmp_list2)
        
        # Assign back to same dictionary or a new one        
        for key in keys:
            dictionary[key][feature] = tmp_list2[n]
            n +=1
        
    #return dictionary     

# Feature list for outlier removal - everyone from features_list_new except poi 
feature_list1 = ['salary', 'bonus', 'total_payments', 'total_stock_value','restricted_stock',
                 'exercised_stock_options','restricted_stock_deferred', 'expenses', 'loan_advances', 'long_term_incentive', 
                   'perc_emails_to_poi','perc_emails_from_poi','perc_emails_shared_poi']
outlier_removal(data_dict2, feature_list1)      

# Check with a particular key - yes! it is done 
data_dict2.get('MCDONALD REBECCA')

# Log transformation as data is skewed 
data_dict3 = copy.deepcopy(data_dict2) 

def log_transform(dictionary, features):
    keys = sorted(dictionary.keys())
    for feature in features:
        n=0
        tmp_list1 = []
        tmp_list2 = []
        for key in keys:
            value = dictionary[key][feature]
            tmp_list1.append( float(value) )
        
        #convert list to numpy array
        arr = np.array(tmp_list1)
        # Take log if array value > 0
        shape = arr.shape
        for x in range(0, shape[0]):
            if arr[x] > 0:
                arr[x] = np.log(arr[x])
        #Convert array back to list 
        tmp_list2 = np.array(arr).tolist()
        #print(tmp_list2)
        
        # Assign back to same dictionary or a new one        
        for key in keys:
            dictionary[key][feature] = tmp_list2[n]
            n +=1
        
    #return dictionary

# Implement feature normalization - sclae b/w 0 and 1 based on min_max
# Create a copy of the dictionary
# Feature list for outlier removal - everyone from features_list_new except poi 
log_transform(data_dict3, feature_list1)
data_dict2.get('HUMPHREY GENE E')
data_dict3.get('HUMPHREY GENE E')

data_dict4 = copy.deepcopy(data_dict3)  

def feature_scaling(dictionary, features):
    keys = sorted(dictionary.keys())
    for feature in features:
        n=0
        tmp_list1 = []
        tmp_list2 = []
        for key in keys:
            value = dictionary[key][feature]
            tmp_list1.append( float(value) )
        #convert list to numpy array
        arr = np.array(tmp_list1)
        min_val = np.amin(arr)
        max_val = np.amax(arr)
        shape = arr.shape
        for x in range(0, shape[0]):
            arr[x] = (arr[x] - min_val)/(max_val - min_val)
        #print(arr)
        #Convert array back to list 
        tmp_list2 = np.array(arr).tolist()
        #print(tmp_list2)
        
        # Assign back to same dictionary or a new one        
        for key in keys:
            dictionary[key][feature] = tmp_list2[n]
            n +=1
        
    #return dictionary 

# Feature list for scaling - everyone except poi      
feature_scaling(data_dict4, feature_list1)  

# Check that has happened
data_dict4.get('MCDONALD REBECCA')
       
### Store to my_dataset for easy export below.
my_dataset = data_dict4

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list_new, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Create features and labels into an array
labels_arr = np.array(labels)
features_arr = np.array(features)
print(labels_arr.shape)
print(features_arr.shape)

## SECTION Below is for plotting and visualization only
# Add labels array to features array
features_arr2 = np.column_stack((features_arr, labels_arr))
print(features_arr2.shape)

# Convert to a dataframe 
import pandas as pd

col_names = ['salary', 'bonus', 'total_payments', 'total_stock_value','restricted_stock',
                 'exercised_stock_options','restricted_stock_deferred', 'expenses', 'loan_advances', 'long_term_incentive', 
                   'perc_emails_to_poi','perc_emails_from_poi','perc_emails_shared_poi', 'poi']
df = pd.DataFrame(features_arr2, columns=col_names)
df.shape
df.head()

sel_feature = 'total_payments'
mean_list = []
mean_list = df.groupby(['poi'])[sel_feature].mean()
print(mean_list)

## Do some plotting 
import matplotlib.pyplot as plt
# Plot Histogram with frequency distribution of selected feature
plt.clf()
plt.hist(df[sel_feature])
plt.title("Feature Distribution")
plt.xlabel("Feature")
plt.ylabel("Frequency")
plt.show()

# Plot bar charts of mean of selected feature by POI - Yes or No 
fig, ax = plt.subplots()
plt.clf()
index = np.arange(2)
bar_width = 0.35
plt.bar(index, mean_list, bar_width)
plt.xlabel('Poi')
plt.ylabel('Means')
plt.title('Means of select feature by POI')
plt.xticks(index + bar_width, ('No', 'Yes'))
plt.show()

# Build intuition - Most of the selected features seem important including the newly created 3 features 

# Split dataset into testing and training set with 80%/20% split
features = features_arr
labels = labels_arr

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2, random_state=40)

# Show the results of the split
print "Training set has {} samples.".format(features_train.shape[0])
print "Testing set has {} samples.".format(features_test.shape[0])

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#Take evaluation function F1 score from code in tester.py
from sklearn.cross_validation import StratifiedShuffleSplit
def f1_calculate(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        return f1
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."


from sklearn.naive_bayes import GaussianNB
clf_A = GaussianNB()
from sklearn.svm import SVC
clf_B = SVC()
from sklearn.tree import DecisionTreeClassifier
clf_C = DecisionTreeClassifier()
from sklearn.neighbors import KNeighborsClassifier
clf_D =  KNeighborsClassifier() # default no_neighbours:5

for clf in (clf_A, clf_B, clf_C, clf_D):
    clf = clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc_score = f1_calculate(clf, my_dataset, features_list_new, folds = 1000)
    print("F1 score =", acc_score)

# Clf_A: ('F1 score =', 0.37422343898073074)
# Clf_B: ('F1 score =', None)
# Clf_C:('F1 score =', 0.2762271414821944)
# CLf_D: ('F1 score =', 0.23222477064220184)

from tester import test_classifier
test_classifier(clf_A, my_dataset, features_list_new, folds = 1000)
# Accuracy: 0.60380       Precision: 0.23703      Recall: 0.88850
test_classifier(clf_B, my_dataset, features_list_new, folds = 1000)
# Precision or recall may be undefined due to a lack of true positive predicitons.
test_classifier(clf_C, my_dataset, features_list_new, folds = 1000)
# Accuracy: 0.79747       Precision: 0.25972      Recall: 0.28050
test_classifier(clf_D, my_dataset, features_list_new, folds = 1000)
# Accuracy: 0.82147       Precision: 0.27218      Recall: 0.20250


# All the classifiers have very different predictions. SVC fails!
# First find optimal no of components
from sklearn.decomposition import PCA
pca = PCA(n_components=10).fit(features_arr)
print(pca.explained_variance_ratio_) 

#[ 0.50922813  0.12955958  0.09989295  0.08667231  0.05500838  0.03398812
#  0.02917271  0.01846822  0.01199314  0.00845632]
# first 5 components explain 88% of the variance 


# Narrow down to 4 to be further tuned with PCA
#1. Guassian NB
#2. Decision Tree
#3. KNN
# Implement PCA using pipeline
# Pipeline on GNB
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB

clf = Pipeline(steps = [('pca', PCA(n_components=6)), ('gnb', GaussianNB())])
clf = clf.fit(features_train, labels_train)
test_classifier(clf, my_dataset, features_list_new, folds = 1000)
# Accuracy: 0.84540       Precision: 0.12993      Recall: 0.02800 (n_comp:4)
# Accuracy: 0.85333       Precision: 0.31550      Recall: 0.08550 (n_comp;5)
# Accuracy: 0.84333       Precision: 0.35772      Recall: 0.22000  (n_comp:6)

# Pipeline on Decision Trees
from sklearn.tree import DecisionTreeClassifier

clf = Pipeline(steps = [('pca', PCA(n_components=4)), ('dt', DecisionTreeClassifier())])
clf = clf.fit(features_train, labels_train)
test_classifier(clf, my_dataset, features_list_new, folds = 1000)
# Accuracy: 0.84827       Precision: 0.41986      Recall: 0.36150 (n_comp; 4)
#  Accuracy: 0.85100       Precision: 0.43251      Recall: 0.37650 (n_comp:5)

# Pipeline on KNN
from sklearn.neighbors import KNeighborsClassifier
clf = Pipeline(steps = [('pca', PCA(n_components=4)), ('knn', KNeighborsClassifier())])
clf = clf.fit(features_train, labels_train)
test_classifier(clf, my_dataset, features_list_new, folds = 1000)
# Accuracy: 0.83380       Precision: 0.06678      Recall: 0.01900 (n_comp;4)
# Accuracy: 0.82553       Precision: 0.10801      Recall: 0.04250 (n-comp;5)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Select Decision Tree classifier for further tuning

# I will tune no of PCA components and max_depth
for n_comp in (3,4,5,6,7): 
    for depth in (2,3,4,5,6,7,8):
        clf = Pipeline(steps = [('pca', PCA(n_components=n_comp)), ('dt', DecisionTreeClassifier(max_depth = depth))])
        clf = clf.fit(features_train, labels_train)
        acc_score = f1_calculate(clf, my_dataset, features_list_new, folds = 1000)
        print("No of PCA components and max_depth =", n_comp, depth)    
        print("F1 score =", acc_score)

#Optimal no of components = 5 and max_depth: 8
clf = Pipeline(steps = [('pca', PCA(n_components= 5)), ('dt', DecisionTreeClassifier(max_depth = 8))])
test_classifier(clf, my_dataset, features_list_new, folds = 1000)
# Accuracy: 0.85027       Precision: 0.42773      Recall: 0.36400

# Tune Guassian NB
# I will tune no of PCA components
for n_comp in (3,4,5,6,7,8,9): 
        clf = Pipeline(steps = [('pca', PCA(n_components=n_comp)), ('gnb', GaussianNB())])
        clf = clf.fit(features_train, labels_train)
        acc_score = f1_calculate(clf, my_dataset, features_list_new, folds = 1000)
        print("No of PCA components =", n_comp)    
        print("F1 score =", acc_score)

#Optimal no of components = 8
clf = Pipeline(steps = [('pca', PCA(n_components= 8)), ('gnb', GaussianNB())])
test_classifier(clf, my_dataset, features_list_new, folds = 1000)
# Accuracy: 0.83467       Precision: 0.38439      Recall: 0.39900


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
clf = Pipeline(steps = [('pca', PCA(n_components= 5)), ('dt', DecisionTreeClassifier(max_depth = 8))])
dump_classifier_and_data(clf, my_dataset, features_list_new)