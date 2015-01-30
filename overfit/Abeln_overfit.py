# Kaggle Overfitting project
# Manipulated by: Brant Abeln
# Generates ~79% agreement
# Uses Pandas!
import pandas as pd
import numpy
from numpy import linalg
import math

# Read in csv
overfit = pd.read_csv('overfitting.csv')

# Split off the training data and test data
train = overfit[overfit.train == 1]
test = overfit[overfit.train == 0]

# trim off the inital information to only contain the variables
train_trim = train.drop(['case_id','train','Target_Practice','Target_Leaderboard','Target_Evaluate'],axis=1)
test_trim = test.drop(['case_id','train','Target_Practice','Target_Leaderboard','Target_Evaluate'],axis=1)

# add a constant term for the intercept variable to be accounted for
train_trim['var_0'] = 1
cols = train_trim.columns.tolist()
cols = cols[-1:] + cols[:-1]
train_trim = train_trim[cols]

test_trim['var_0'] = 1
cols = test_trim.columns.tolist()
cols = cols[-1:] + cols[:-1]
test_trim = test_trim[cols]

# Keep only the data in numpy arrays
trainarray = numpy.array(train_trim)
testarray = numpy.array(test_trim)

# Make the training practice array
train_practice_array = numpy.array(train["Target_Practice"])
test_practice_array = numpy.array(test["Target_Practice"])

# Least Squares solution
# Theta is the coefficent matrix which will be used for the starting values
theta = numpy.dot(linalg.inv(numpy.dot(numpy.transpose(trainarray),trainarray)),numpy.dot(numpy.transpose(trainarray),train_practice_array))


# define the Heaviside Theta Function
def ftheta(x):
    if x>0:
        return 1
    else:
        return 0

# Stochastic gradient descent method
# We must shuffle our sampling of the pairs of (x_i,y_i) so as not to introduce circular loops on the variables
q=numpy.array(range(len(trainarray)))
numpy.random.shuffle(q)

# Define some weighting function
alpha = 0.10

# Run through the 20 random pairs of functions
# Each run through, optimize all of the theata to each function.
# The thetha correction is the difference in the training value and the functional value of the data
#    weighted by the value of the variable.
# Notice that we shifted the data by 1/2.  It was suggested in the forums.
for k in range(500):
    for i in q[0:20]:
        factor = numpy.dot(theta,(trainarray[i]-0.5))
        for j in range(201):
            # Test of other updating schemes
#            theta[j] = theta[j] - alpha*(train_practice_array[i] - 1.0/(1.0+factor))*trainarray[i][j]*factor/(1.0+factor)/(1.0+factor)
#            theta[j] = theta[j] - 1.0/(trainarray[i][j]*factor/(1.0+factor))
            theta[j] = theta[j] + alpha*(train_practice_array[i] - ftheta(factor))*trainarray[i][j]
    numpy.random.shuffle(q)

# Create a results variable
result=[]

#  Generate the predictions and store in the Results Array
for i in range(len(testarray)):
    factor = numpy.dot(theta,(testarray[i]-0.5))
    result.append(ftheta(factor))

# Calculate the difference in values from the test values and the predicted values
sum = 0
for i in range(len(testarray)):
    sum=sum+abs(test_practice_array[i]-result[i])

print("We guessed", (len(testarray) - sum)/len(testarray)*100, "percent correctly!")
