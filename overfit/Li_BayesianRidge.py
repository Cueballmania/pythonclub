import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn import linear_model

#Step 1: perform logistic regression on the training data
df=pd.read_csv("../Data/overfitting.csv",header=0)

#split data into training and test
df_train=df[df["train"]==1]
df_test=df[df["train"]==0]

#get the data and target for both the training and testing groups
data_train=np.array(df_train.drop(["case_id","train","Target_Practice","Target_Leaderboard","Target_Evaluate"],axis
=1))
target_train=np.array(df_train["Target_Practice"])
data_test=np.array(df_test.drop(["case_id","train","Target_Practice","Target_Leaderboard","Target_Evaluate"],axis=1
))
target_test=np.array(df_test["Target_Practice"])

#BayensianRidge
#See, e.g. http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
clf=linear_model.BayesianRidge()
clf.fit(data_train,target_train)


#step 2: test the accuracy of the fitting function on both the training and the testing data
result_train=np.rint((clf.predict(data_train))).astype(int)
n=len(data_train)
m=np.sum(target_train==result_train)
print "ahccuracy rate in training=",m,n

result_test=np.rint(clf.predict(data_test)).astype(int)
n=len(data_test)
m=np.sum(target_test==result_test)
print "accuracy rate in test=",m,n,float(m)/n
