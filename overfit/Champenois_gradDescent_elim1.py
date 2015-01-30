import numpy as np
import pandas as pd
import pylab as py

def h(m,sampleVars):

    return 1/(1+np.exp(-np.dot(m,(sampleVars-0.5).T)))
    
def J(m,sampleVars,sampleVals):

    hs = h(m,sampleVars)
    
    normalJ = -sum(sampleVals*np.nan_to_num(np.log(hs))+(1-sampleVals)*np.nan_to_num(np.log(1-hs)))
    
    #Jreg = l*sum(m**p)
    
    return normalJ#+Jreg
    
def dJdm(m,sampleVars,sampleVals):
    
    hs = h(m,sampleVars)
    
    normaldJdm = np.dot((hs-sampleVals),sampleVars-0.5)
    
    #dJdmreg = p*l*m**(p-1)
    
    return normaldJdm#+dJdmreg
    
def d2Jdm2(m,sampleVars,sampleVals):
    
    hs = h(m,sampleVars)
    
    normald2Jdm2 = np.dot((1-hs)*hs,(sampleVars-0.5)**2)

    return normald2Jdm2
    
def checkPrediction(m,sampleVars,sampleVals):
    
    return np.sum(sampleVals == np.round(h(m,sampleVars)))/float(sampleVars.shape[0])
    
def newtons(m,alpha,maxIter,fitVars,fitVals):
    
    dJdms = dJdm(m,fitVars,fitVals)
    iter = 0
    while iter<maxIter:
        iter+=1
        m = m - alpha*dJdms
        dJdms = dJdm(m,fitVars,fitVals)
    
    return m,checkPrediction(m,fitVars,fitVals)
    
raw = pd.read_csv('/Users/elio/Documents/Python/overfitting.csv')

sampleVals = raw.Target_Practice[raw.train==1]
varColumns = ['var_'+str(varnum) for varnum in range(1,201)]
sampleVars = raw[varColumns][raw.train==1]
otherVals = raw.Target_Practice[raw.train==0]
otherVars = raw[varColumns][raw.train==0]
    
#m = 1*(2*np.random.rand(len(varColumns))-1)
m=np.dot(np.linalg.inv(np.dot(sampleVars.T,sampleVars)),np.dot(sampleVars.T,sampleVals))

alpha = 1
maxIter = 5

(m,fitPred) = newtons(m,alpha,maxIter,sampleVars,sampleVals)

#%%
def elimVars(m,sampleVars,sampleVals):
    keptVars = range(0,len(m))
    mFilt = mFilt = [m[i] for i in keptVars]
    alpha = 1
    maxIter = 5
    while any(m>0):
        ms = []
        for i in keptVars:
            ms.append(m[i])
        minIndex = ms.index(np.max(ms))
        m[keptVars[minIndex]]=0
        keptVars.pop(minIndex)
        mFilt = [m[i] for i in keptVars]
        varColumnsFilt = ['var_'+str(varnum+1) for varnum in keptVars]
        sampleVarsFilt = raw[varColumnsFilt][raw.train==1]
        mFilt = newtons(mFilt,alpha,maxIter,sampleVarsFilt,sampleVals)[0]
        m = np.zeros(200)
        for i in range(0,len(keptVars)):
            m[keptVars[i]] = mFilt[i]
    return m, checkPrediction(m,sampleVars,sampleVals),checkPrediction(m,otherVars,otherVals)
        
print elimVars(m,sampleVars,sampleVals)[1::]

#%%

