import numpy as np
import pandas as pd

# used in cost function
def h(m,sampleVars):

    return 1/(1+np.exp(-np.dot(m,(sampleVars-0.5).T)))
    
# cost function
def J(m,sampleVars,sampleVals):

    hs = h(m,sampleVars)
    
    return -sum(sampleVals*np.nan_to_num(np.log(hs))+(1-sampleVals)*np.nan_to_num(np.log(1-hs)))
    
# derivative of cost function
def dJdm(m,sampleVars,sampleVals):
    
    hs = h(m,sampleVars)
    
    return np.dot((hs-sampleVals),sampleVars-0.5)

# second derivative of cost function 
def d2Jdm2(m,sampleVars,sampleVals):
    
    hs = h(m,sampleVars)

    return np.dot((1-hs)*hs,(sampleVars-0.5)**2)

# check predictive power of m
def checkPrediction(m,Vars,Vals):
    
    return np.sum(Vals == np.round(h(m,Vars)))/float(len(Vals))
    
# simple gradient descent optimizer
def newtons(m,alpha,maxIter,fitVars,fitVals):
    
    dJdms = dJdm(m,fitVars,fitVals)
    iter = 0
    while iter<maxIter:
        iter+=1
        m = m - alpha*dJdms
        dJdms = dJdm(m,fitVars,fitVals)
    
    return m,checkPrediction(m,fitVars,fitVals)

# read in data
raw = pd.read_csv('/Users/elio/Documents/Python/overfitting.csv')

# parse data
sampleVals = np.array(raw.Target_Practice[raw.train==1])
varColumns = ['var_'+str(varnum) for varnum in range(1,201)]
sampleVars = np.array(raw[varColumns][raw.train==1])
otherVals = raw.Target_Practice[raw.train==0]
otherVars = np.array(raw[varColumns][raw.train==0])

# initial guess for fit, random works fine, but using the least squares basic fit gives a robust start (found from Brant's code)
#m = 1*(2*np.random.rand(len(varColumns))-1)
m=np.dot(np.linalg.inv(np.dot(sampleVars.T,sampleVars)),np.dot(sampleVars.T,sampleVals))

# played with gradient descent parameters and these seem to be good enough and fast
alpha = 1
maxIter = 5

# optimize the initial guess
(m,fitPred) = newtons(m,alpha,maxIter,sampleVars,sampleVals)

# function to eliminate variables and optimize fits
def elimVars(m,sampleVars,sampleVals):
    
    # initialize some variables
    keptVars = range(0,len(m)) # indices of variables kept
    mFilt = [m[i] for i in keptVars] # m filtered for kept variables
    alpha = 1
    maxIter = 5
    
    # remove one positive parameter at a time until none left
    while any(m>0):
        ms = []
        for i in keptVars:
            ms.append(m[i])
        minIndex = ms.index(np.max(ms)) # find index of largest m
        m[keptVars[minIndex]]=0 # set it to 0
        keptVars.pop(minIndex) # remove index from keptVars
        mFilt = [m[i] for i in keptVars] # repopulate mFilt
        sampleVarsFilt = sampleVars[:,keptVars] # remove unwated observed variables
        mFilt = newtons(mFilt,alpha,maxIter,sampleVarsFilt,sampleVals)[0] # optimize remaining parameters
        m = m*0 # repopulate m
        for i in range(0,len(keptVars)):
            m[keptVars[i]] = mFilt[i]
    return m

# run the function
mFinal = elimVars(m,sampleVars,sampleVals)

# this gives ~83.5% accuracy
print checkPrediction(mFinal,sampleVars,sampleVals),checkPrediction(mFinal,otherVars,otherVals)

# Trying some "bootstrapping" - running different optimizations to see which variables stick around more often than others
# Didn't help over previous method

Nsets = 200 # number of new data sets
bootstrap = np.random.randint(0,len(sampleVals),(250,Nsets)) # see which observations to put into each new data set
mMat = np.zeros((Nsets,len(m))) # initialize parameter fits

# loop over data sets, find which variables are kept for each one
for i in range(0,Nsets):
    
    # make new data set
    resampledVars = sampleVars[bootstrap[:,i],:]
    resampledVals = sampleVals[bootstrap[:,i]]
    
    # find initial guess for m
    resampledM = np.dot(np.linalg.inv(np.dot(resampledVars.T,resampledVars)),np.dot(resampledVars.T,resampledVals))
    # optimize this initial guess    
    resampledM = newtons(resampledM,alpha,maxIter,resampledVars,resampledVals)[0]  
    # reoptimize while getting rid of variables
    mMat[i,:] = elimVars(resampledM,resampledVars,resampledVals)

print checkPrediction(m,otherVars,otherVals)
# try to keep a certain number of variables that survived most often in the previous step and optimize again
sortInds = np.argsort(sum(mMat==0)) # look at which variables get kept the most
for numMs in range(50,200): # loop over how many variables to keep

    subM = m[sortInds[0:numMs]] # keep best variables from initial guess
    subM = newtons(subM,alpha,maxIter,sampleVars[:,sortInds[0:numMs]],sampleVals)[0]
    subM = elimVars(subM,sampleVars[:,sortInds[0:numMs]],sampleVals)
    print numMs,checkPrediction(subM,otherVars[:,sortInds[0:numMs]],otherVals)
