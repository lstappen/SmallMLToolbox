# Manual Confusion Matrix + Cross Data split + Mean/Standard deviation
# without packages

# Call: 
# 	getCFMatrix( X_test, y_test, model, prediction )
#   model: e.g. sklearn tree
def getCFMatrix( self, test, target, model, prediction):
	# initialise counters
	TP = 0 # true positives
	TN = 0 # true negatives
	FP = 0 # false positive
	FN = 0 # false negatives
	#Compare X_test prediction against real classes
	for i in range(len(test)):
	  pred = []
	  if len(prediction) >= 1:
		  pred = prediction
		  j = i
	  else:
		  pred = model.predict(test[i].reshape(1,-1))
		  j = 0
	  if pred[j] == 1 and target[i] == 1: # True positive
		  TP += 1
	  if pred[j] == 1 and target[i] == 0: # False positive
		  FP += 1
	  if pred[j] == 0 and target[i] == 1: # False negative
		  FN += 1
	  if pred[j] == 0 and target[i] == 0: # True negative
		  TN += 1		  
	CFM = [[0 for x in range(2)] for y in range(2)]
	CFM[0][0] = TP
	CFM[0][1] = FP
	CFM[1][0] = FN
	CFM[1][1] = TN
	#Basic CFM calculations: Precision, Recall, F1 score
	precision = TP / (TP + FP)
	recall = TP / (TP + FN)
	f1Score = 2 * ((precision * recall)/(precision + recall))
	return (CFM,precision,recall,f1Score)


#Ratio is the size of test in %
def getCrossTrainTestSplit ( self, X, y , ratio):
	noRecords = len(y) #or X[:1] 
	noTestRecords = math.ceil( noRecords / (100*ratio))
	# sampling test by random numbers
	idxTestRows = rnd.sample(range(0, int(noRecords)), int(noTestRecords))
	# Sampling train by minus amount
	idxTrainRows = set(range(0, noRecords)) - set(idxTestRows)
	# Select train and test
	X_train = X[list(idxTrainRows),:]
	X_test  = X[idxTestRows,:]
	y_train = y[list(idxTrainRows)]
	y_test  = y[idxTestRows]
	return (X_train, X_test, y_train, y_test)


import math
def getStats ( data):
    meanS = (1/len(data))*sum(data)
    stdDS = math.sqrt( (1/(len(data)-1))*sum([pow(i-meanS,2) for i in data]) )
    return ( [meanS, stdDS])
