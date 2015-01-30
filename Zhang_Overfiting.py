import pandas as pd
import numpy as np
import time as time
from operator import itemgetter

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('../csv/overfitting.csv', header=0)
print "Finish reading"
#df_train=df_train.replace([-99],['NaN'])
#df_train.dropna()
TrainSize=250
VarNum=200
TotSize=20000
TestSize=TotSize-TrainSize
df_train=df.iloc[0:TrainSize,:]
df_test=df.iloc[TrainSize:TotSize+1,:]
DataTrain=df_train.as_matrix()
DataTest =df_test.as_matrix()

Target_Practice=DataTrain[:,2]
DataTrain=DataTrain[:,5:VarNum+6]
Target_Practice_Test=DataTest[:,2]
DataTest =DataTest[:,5:VarNum+6]


"""
U,s,V=np.linalg.svd(DataTrain,full_matrices=True)
print U
var=raw_input("U")
print s
var=raw_input("s")
print V
var=raw_input("V")
"""

Target_Practice=np.array(Target_Practice)
DataTrain=np.array(DataTrain)
from sklearn import svm
from sklearn import tree
from sklearn import cross_validation
from sklearn import metrics
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

logreg=linear_model.LogisticRegression(C=1e5)


def LogisticRegressionAndReOrder(DataTrain,Target_Practice,DataTest):
	X,y=DataTrain,Target_Practice
	X=StandardScaler().fit_transform(X)
	logreg.fit(X,y)
	VarNum_now=len(DataTrain[0,:])
	sort_coef_=[[0.0 for x in range(2)] for x in range(VarNum_now)]
	sort_coef_=np.array(sort_coef_)
	sort_coef_[:,0]=list(range(0,VarNum_now))
	sort_coef_[:,1]=logreg.coef_[0,:]
	coef_beforeSort_=logreg.coef_[0,:]
	sort_coef_=sorted(abs(sort_coef_), reverse=True,key=itemgetter(1))
	DataTrain_Reorder=[[0.0 for x in range(VarNum_now)] for x in range(TrainSize)]
	DataTrain_Reorder=np.array(DataTrain_Reorder)
	DataTest_Reorder=[[0.0 for x in range(VarNum_now)] for x in range(TestSize)]
	DataTest_Reorder=np.array(DataTest_Reorder)
	for i in range(VarNum_now):
		DataTrain_Reorder[:,i]=DataTrain[:,sort_coef_[i][0].astype(np.int)]
		DataTest_Reorder[:,i] =DataTest[:,sort_coef_[i][0].astype(np.int)]
	return coef_beforeSort_,DataTrain_Reorder,sort_coef_,DataTest_Reorder

#coef_,DataTrain_Reorder,sort_coef_=LogisticRegressionAndReOrder(DataTrain,Target_Practice)	

def LogisticRegressionWithOneFewerDimension(DataTrain,Target_Practice,DataTest):
	VarNum_pre=len(DataTrain[0,:])
	VarNum_now=VarNum_pre-1
	DataTrain_Reduced=DataTrain[:,0:VarNum_now]
	DataTest_Reduced =DataTest[:,0:VarNum_now]
	X,y=DataTrain_Reduced,Target_Practice
	X=StandardScaler().fit_transform(X)
	logreg.fit(X,y)
	coef_beforeSort_=logreg.coef_[0,:]
	return coef_beforeSort_,DataTrain_Reduced,DataTest_Reduced

def ReduceDimensionOneByOne(DataTrain,Target_Practice,DataTest):
	coef_,DataTrain_Reorder,sort_coef_ ,DataTest_Reorder  =LogisticRegressionAndReOrder(DataTrain,Target_Practice,DataTest)
	coef_beforeSort_,DataTrain_Reduced, DataTest_Reduced=LogisticRegressionWithOneFewerDimension(DataTrain_Reorder,Target_Practice,DataTest_Reorder)
	coef2_,DataTrain_Reorder2,sort_coef2_,DataTest_Reorder2=LogisticRegressionAndReOrder(DataTrain_Reduced,Target_Practice,DataTest_Reduced)
	Theta_cut=(sort_coef2_[len(sort_coef2_)-1][1])
	print "theta_cut",Theta_cut
	i=0
	while(abs(Theta_cut)<1):
		DataTrain_Reorder=[]
		DataTrain_Reorder=DataTrain_Reorder2
		coef_beforeSort_=[]
		DataTrain_Reduced=[]
		DataTest_Reduced=[]
		coef_beforeSort_,DataTrain_Reduced,DataTest_Reduced=LogisticRegressionWithOneFewerDimension(DataTrain_Reorder,Target_Practice,DataTest_Reorder)
		coef2_=[]
		DataTrain_Reorder2=[]
		sort_coef2_=[]
		DataTest_Reorder2=[]
		coef2_,DataTrain_Reorder2,sort_coef2_,DataTest_Reorder2=LogisticRegressionAndReOrder(DataTrain_Reduced,Target_Practice,DataTest_Reduced)
		del Theta_cut
		Theta_cut=(sort_coef2_[len(sort_coef2_)-1][1])
		i=i+1
		print Theta_cut,i
	return coef2_,DataTrain_Reorder2,sort_coef2_,DataTest_Reorder2

coef_final_,DataTrain_Reorder_final,sort_coef_final_,DataTest_Reorder_final=ReduceDimensionOneByOne(DataTrain,Target_Practice,DataTest)

def FeaturePower(Data,power):
	VarNum=len(Data[0,:])
	DataSize=len(Data)
	Data_Power=[[0.0 for x in range(VarNum*2)] for x in range(DataSize)]
	Data_Power=np.array(Data_Power)
	for i in range(VarNum):
		Data_Power[:,i]=Data[:,i]
	for i in range(VarNum):
		Data_Power[:,VarNum+i]=np.power(Data_Power[:,i],power)
	return Data_Power


def DirectLogisticPrediction(DataTrain,Target_Practice,DataTest,Target_Practice_Test):
	X,y=DataTrain,Target_Practice
	X=StandardScaler().fit_transform(X)
	logreg.fit(X,y)
	y_t=logreg.predict(DataTest)
	score1=logreg.score(X,y)
	score2=logreg.score(DataTest,Target_Practice_Test)
	return score1, score2


X,y=DataTrain_Reorder_final,Target_Practice
X=StandardScaler().fit_transform(X)
X_T,y_T=DataTest_Reorder_final,Target_Practice_Test
X_T=StandardScaler().fit_transform(X_T)

X_BT=DataTrain
X_BT=StandardScaler().fit_transform(X_BT)
X_T_BT=DataTest
X_T_BT=StandardScaler().fit_transform(X_T_BT)

X_P2=FeaturePower(X,2)
X_P2=StandardScaler().fit_transform(X_P2)
X_T_P2=FeaturePower(X_T,2)
X_T_P2=StandardScaler().fit_transform(X_T_P2)

ScoreData=[]
names_RBF=[
##"RBF gamma=1,C=1",  "RBF gamma=0.5, C=1", "RBF gamma=0.1, C=1", "RBF gamma=0.05, C=1",
##	#"RBF gamma=1,C=0.5", "RBF gamma=0.5, C=0.5", "RBF gamma=0.1, C=0.5", "RBF gamma=0.05, C=0.5",
##	#"RBF gamma=1,C=2", "RBF gamma=0.5, C=2", "RBF gamma=0.1, C=2", "RBF gamma=0.05, C=2"
	"RBF gamma=0.01,C=1",  "RBF gamma=0.005, C=1", "RBF gamma=0.001, C=1",
	"RBF gamma=0.01,C=2",  "RBF gamma=0.005, C=2", "RBF gamma=0.001, C=2",
	"RBF gamma=0.01,C=4",  "RBF gamma=0.005, C=4", "RBF gamma=0.001, C=4",
	"RBF gamma=0.01,C=8",  "RBF gamma=0.005, C=8", "RBF gamma=0.001, C=8",
	"RBF gamma=0.01,C=16",  "RBF gamma=0.005, C=16", "RBF gamma=0.001, C=16",
	"RBF gamma=0.01,C=32",  "RBF gamma=0.005, C=32", "RBF gamma=0.001, C=32",
	"RBF gamma=0.01,C=64",  "RBF gamma=0.005, C=64", "RBF gamma=0.001, C=64",
#	"RBF gamma=0.01,C=0.5",  "RBF gamma=0.005, C=0.5", "RBF gamma=0.001, C=0.5",
#	# "RBF gamma=0.0005, C=1"
	]
classifiers_RBF=[
##svm.SVC(kernel="rbf", gamma=1,C=1),
##		svm.SVC(kernel="rbf", gamma=0.5,C=1),
##		svm.SVC(kernel="rbf", gamma=0.1,C=1),
##		svm.SVC(kernel="rbf", gamma=0.05,C=1),
###		svm.SVC(kernel="rbf", gamma=1,C=0.5),
###               svm.SVC(kernel="rbf", gamma=0.5,C=0.5),
###              svm.SVC(kernel="rbf", gamma=0.1,C=0.5),
###             svm.SVC(kernel="rbf", gamma=0.05,C=0.5),
###		svm.SVC(kernel="rbf", gamma=1,C=2),
###		svm.SVC(kernel="rbf", gamma=0.5,C=2),
###		svm.SVC(kernel="rbf", gamma=0.1,C=2),
###		svm.SVC(kernel="rbf", gamma=0.05,C=2)
                svm.SVC(kernel="rbf", gamma=0.01,C=1),
		svm.SVC(kernel="rbf", gamma=0.005,C=1),
		svm.SVC(kernel="rbf", gamma=0.001,C=1),
                svm.SVC(kernel="rbf", gamma=0.01,C=2),
		svm.SVC(kernel="rbf", gamma=0.005,C=2),
		svm.SVC(kernel="rbf", gamma=0.001,C=2),
                svm.SVC(kernel="rbf", gamma=0.01,C=4),
		svm.SVC(kernel="rbf", gamma=0.005,C=4),
		svm.SVC(kernel="rbf", gamma=0.001,C=4),
                svm.SVC(kernel="rbf", gamma=0.01,C=8),
		svm.SVC(kernel="rbf", gamma=0.005,C=8),
		svm.SVC(kernel="rbf", gamma=0.001,C=8),
                svm.SVC(kernel="rbf", gamma=0.01,C=16),
		svm.SVC(kernel="rbf", gamma=0.005,C=16),
		svm.SVC(kernel="rbf", gamma=0.001,C=16),
                svm.SVC(kernel="rbf", gamma=0.01,C=32),
		svm.SVC(kernel="rbf", gamma=0.005,C=32),
		svm.SVC(kernel="rbf", gamma=0.001,C=32),
                svm.SVC(kernel="rbf", gamma=0.01,C=64),
		svm.SVC(kernel="rbf", gamma=0.005,C=64),
		svm.SVC(kernel="rbf", gamma=0.001,C=64),
#                svm.SVC(kernel="rbf", gamma=0.01,C=0.5),
#		svm.SVC(kernel="rbf", gamma=0.005,C=0.5),
#		svm.SVC(kernel="rbf", gamma=0.001,C=0.5),
#		svm.SVC(kernel="rbf", gamma=0.0005,C=1)
		]

for name, clf in zip(names_RBF, classifiers_RBF):
	clf.fit(X,y)
	score1=clf.score(X,y)
	score2=clf.score(X_T,y_T)
	print name,"after trim", score1,score2
	clf.fit(X_BT,y)
	s1=clf.score(X_BT,y)
	s2=clf.score(X_T_BT,y_T)
	print name,"before trim",s1,s2
	ScoreData.append([name,score1,score2,s1,s2])
	print "   "

import csv
with open('RBF_score.csv','wb') as ScoreData:
	writer=csv.writer(ScoreData)
	for row in ScoreData:
		writer.writerows(row)
raw_input("xxx")

gnb=GaussianNB()
gnb.fit(X,y)
score1=gnb.score(X_T,y_T)
score2=gnb.score(X,y)
print "after trim Gaissuan NB on test=",score1," Gaussian NB on train=",score2
gnb.fit(X_BT,y)
score1=gnb.score(X_T_BT,y_T)
score2=gnb.score(X_BT,y)
print "before trim Gaussian NB on test",score1, " Gaussian NB on train",score2
raw_input("xxxx")

s1,s2=DirectLogisticPrediction(X,y,X_T,y_T)
print "after trim train=",s1,"test=",s2
s1,s2=DirectLogisticPrediction(X_P2,y,X_T_P2,y_T)
print "power 2 after trim train=",s1,"test=",s2
s1,s2=DirectLogisticPrediction(X_BT,y,X_T_BT,y_T)
print "before trim train=",s1,"test=",s2
raw_input("xxxx")


names_PolySVM=["Poly SVM 1","Poly SVM 2","Poly SVM 3", "Poly SVM 4", "Poly SVM 5"]
classifiers_PolySVM=[
	svm.SVC(kernel="poly",degree=1),
	svm.SVC(kernel="poly",degree=2),
	svm.SVC(kernel="poly"),
	svm.SVC(kernel="poly",degree=4),
	svm.SVC(kernel="poly",degree=5)]

for name, clf in zip(names_PolySVM, classifiers_PolySVM):
	clf.fit(X,y)
	score1=clf.score(X,y)
	score2=clf.score(X_T,y_T)
	print name,"after trim", score1,score2
	clf.fit(X_P2,y)
	score1=clf.score(X_P2,y)
	score2=clf.score(X_T_P2,y_T)
	print name,"power 2 after trim", score1,score2
	clf.fit(X_BT,y)
	score1=clf.score(X_BT,y)
	score2=clf.score(X_T_BT,y_T)
	print name,"before trim",score1,score2

raw_input("xxx")


names_SVM=["linear SVM","Poly SVM", "RBF SVM"]
classifiers_SVM=[
	svm.SVC(kernel="linear", C=0.025),
	svm.SVC(kernel="poly"),
	svm.SVC()]

for name, clf in zip(names_SVM, classifiers_SVM):
	clf.fit(X,y)
	score1=clf.score(X,y)
	score2=clf.score(X_T,y_T)
	print name,"after trim", score1,score2
	clf.fit(X_BT,y)
	score1=clf.score(X_BT,y)
	score2=clf.score(X_T_BT,y_T)
	print name,"before trim",score1,score2

raw_input("xxx")
	

names = ["Nearest Neighbors 3", "Linear SVM 0.025", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    svm.SVC(kernel="linear", C=0.025),
    svm.SVC(gamma=2, C=1),
    tree.DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]


names_linear=["C=0.025","C=0.25","C=0.5","C=0.75","C=1"]
classifiers_linear=[svm.SVC(kernel="linear", C=0.025),
	svm.SVC(kernel="linear", C=0.25),
	svm.SVC(kernel="linear", C=0.50),
	svm.SVC(kernel="linear", C=0.75),
	svm.SVC(kernel="linear", C=1)]
for name, clf in zip(names_linear, classifiers_linear):
	clf.fit(X,y)
	score1=clf.score(X,y)
	score2=clf.score(X_T,y_T)
	print name,"after trim", score1,score2
	clf.fit(X_BT,y)
	score1=clf.score(X_BT,y)
	score2=clf.score(X_T_BT,y_T)
	print name,"before trim",score1,score2

import scipy
from scipy.stats import pearsonr
import pylab as P
import matplotlib.pyplot as plt 


