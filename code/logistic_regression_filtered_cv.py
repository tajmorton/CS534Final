#!/usr/bin/python
## Feature Selector
import sys
import Orange, orange, orngTest, orngStat
import proj_utils
import os


#based on http://orange.biolab.si/doc/ofb/c_performance.htm
#def accuracy(test_data, classifiers): 
#    correct = [0.0]*len(classifiers) 
#    for ex in test_data: 
#        for i in range(len(classifiers)): 
#            if classifiers[i](ex) == ex.getclass(): 
#                correct[i] += 1 
#    for i in range(len(correct)): 
#        correct[i] = correct[i] / len(test_data) 
#    return correct 
"""
!!!WARNING!!!
right now feature selection occurs separately with each validation set's classification, 
so it's doing feature selection on the subset and perhaps retrieving a feature incompatible
with regression in the process, so it throws an exception.  I'm not sure how to change this right now.
"""

"""
Trains Logistic Regression with selected features
"""
def train_classifier(data, features):
    logr = Orange.classification.logreg.LogRegLearner()
    #learner = Orange.feature.selection.FilteredLearner(logr, filter=Orange.feature.selection.FilterBestN(n=features), name='filtered')
    logr_classifier = logr(data)
    return logr_classifier

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print >> sys.stderr, "Usage: logistic_regression_filtered_cv.py [TAB_FILE] [NUM_FOLDS] [NUM_FEATURES]"
        sys.exit(1)

    start_data = proj_utils.load_data(sys.argv[1])

    cv_folds = int(sys.argv[2])
    features = int(sys.argv[3])
    
    # default scoring algorithm
    scores = Orange.feature.scoring.score_all(start_data)
    data = Orange.feature.selection.select(start_data, scores, features)
    
    train_data, test_data = proj_utils.partition_data(data)
    
    selection = orange.MakeRandomIndicesCV(data, cv_folds)

    sen1 = 0.0
    spe1 = 0.0
    acc1 = 0.0
    sen2 = 0.0
    spe2 = 0.0
    acc2 = 0.0
    
 
    
    """
    based on http://orange.biolab.si/doc/ofb/c_performance.htm
    """
    for test_fold in range(cv_folds):
        train_data = data.select(selection, test_fold, negate=1)
        test_data = data.select(selection, test_fold)
        
        model = train_classifier(train_data, features)
        
        train_CA, train_results = proj_utils.test_classifier(model, train_data)
        test_CA, test_results = proj_utils.test_classifier(model, test_data)        
        
        train_stats = proj_utils.get_stats(train_results)
        test_stats = proj_utils.get_stats(test_results)        

        #acc1 += accuracy(test_data, model)
        #sen1 += sensitivity(test_data, model)
        #spe1 += specificity(test_data, model)
        sen1 += train_stats.get('Sensitivity')
        spe1 += train_stats.get('Specificity')
        acc1 += train_stats.get('Accuracy')
        sen2 += test_stats.get('Sensitivity')
        spe2 += test_stats.get('Specificity')
        acc2 += test_stats.get('Accuracy')       
        
        print "Training set %d: %s %s %s\n" % (test_fold+1, sen1/(test_fold+1), spe1/(test_fold+1), acc1/(test_fold+1))
        print "Test set %d: %s %s %s\n" % (test_fold+1, sen2/(test_fold+1), spe2/(test_fold+1), acc2/(test_fold+1))
    acc1 /= cv_folds
    sen1 /= cv_folds
    spe1 /= cv_folds
    acc2 /= cv_folds
    sen2 /= cv_folds
    spe2 /= cv_folds    
        
    train_CA, train_results = proj_utils.test_classifier(model, train_data)
    test_CA, test_results = proj_utils.test_classifier(model, test_data)

    #print "Train Accuracy: %f, Test Accuracy: %f" % (train_CA, test_CA)


    print "Train:\n Sensitivity: %s, Specificity: %s, Accuracy: %s\n" % (str(sen1), str(spe1), str(acc1))
    print "Test:\n Sensitivity: %s, Specificity: %s, Accuracy: %s\n" % (str(sen2), str(spe2), str(acc2))
    
    f = open(os.path.dirname(__file__) + '\\logisticRegressionFilteredCVResults' + str(features) + '.txt', 'w+')
    f.write("Train:\n Sensitivity: %s, Specificity: %s, Accuracy: %s\n" % (str(sen1), str(spe1), str(acc1)))
    f.write("Test:\n Sensitivity: %s, Specificity: %s, Accuracy: %s\n" % (str(sen2), str(spe2), str(acc2)))   
    f.close()
