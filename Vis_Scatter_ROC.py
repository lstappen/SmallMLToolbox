
# Create a scatter plot for every feature combination with feature name description
# and color map for class
#feature_num is number of features e.g from header
plt.figure(figsize=(15,15))
plt.rc( 'xtick', labelsize=8 )
plt.rc( 'ytick', labelsize=8 )
for x in range( 0, feature_num ):
    for y in range( x+1,feature_num ):
        plt.subplot( feature_num, feature_num, y*feature_num+x+1 )
        plt.scatter( data[:,x], data[:,y], c=labels4.astype(np.float) )
        plt.xlabel( feature_names[x], fontsize=8 )
        plt.ylabel( feature_names[y], fontsize=8 )
        plt.axis( "tight" )
plt.show()


#Smooth ROC curve
# Initialise list for tp and fp
TPList = []
FPList = []
model_tmp = [] # variable used in the class
for j in range(500): # test 500 different thresholds
    threshold = 0.01 * j
    tPredict = []
    # tPredict = list of predictions
    # pPredict = set of class probabilities
	# For a given threshold, we can turn this into a list of class predictions. 
    # AUC is related to the probability that the classifier will correctly 
	# classify a randomly chosen example
    for i in range(len(X_test)):
        if pPredict[i,0] >= threshold: # above the threshold
           tPredict.append(0)
        else:                              # below the threshold
           tPredict.append(1)
		   
    # Get the confusion matrix elements e.g. from manual 
    cfM = getCFMatrix( X_test, y_test, model_tmp, tPredict ) # get matrix
    #Add TP, FP
    TPList.append(cfM[0][0])
    FPList.append(cfM[0][1])

# Normalise the tp and fp lists in percentual terms:
TPList = [(100*(i/max(TPList))) for i in TPList]
FPList = [(100*(i/max(FPList))) for i in FPList]
#Plot
plt.plot(FPList, TPList, '-o')
plt.xlabel('FP rate')
plt.ylabel('TP rate')
plt.axis([-10, 110, -10, 110])
plt.show()
