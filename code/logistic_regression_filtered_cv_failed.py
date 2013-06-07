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

    #whole_table = proj_utils.load_data(sys.argv[1])
    #start_domain = Orange.data.Domain(whole_table.domain.attributes[4:])
    #start_data = Orange.data.Table(start_domain, whole_table)
    #example_table = orange.ExampleTable(sys.argv[1])
    #example_start_domain = Orange.data.Domain(example_table.domain.attributes[4:])
    #example_data = orange.ExampleTable(example_start_domain, example_table)
    
    start_data = proj_utils.load_data(sys.argv[1])
    example_data = orange.ExampleTable(sys.argv[1])

    cv_folds = int(sys.argv[2])
    features = int(sys.argv[3])
    
    # default scoring algorithm
    #scores = Orange.feature.scoring.score_all(start_data)
    #data = Orange.feature.selection.select(start_data, scores, features)
    
    train_data, test_data = proj_utils.partition_data(start_data)
    
    #selection = orange.MakeRandomIndicesCV(data, cv_folds)

    #sen1 = 0.0
    #spe1 = 0.0
    #acc1 = 0.0
    #sen2 = 0.0
    #spe2 = 0.0
    #acc2 = 0.0
    
    
    model = train_classifier(train_data, features)    
    train_results = orngTest.crossValidation([model], example_data, cv_folds)
    #test_results = orngTest.crossValidation([model], test_data, cv_folds)    
    
    train_stats = proj_utils.get_stats(train_results)
    #test_stats = proj_utils.get_stats(test_results)       

    print "Train:\n%s" % str(train_stats)
    #print "\nTest:\n%s" % str(test_stats)
    
    f = open(os.path.dirname(__file__) + '\\logisticRegressionFilteredCVResults_' + 'V' + str(cv_folds) + '_F' + str(features) + '.txt', 'w+')
    f.write("Train:\n")
    f.write(str(train_stats) + "\n")
    #f.write("Test:\n")
    #f.write(str(test_stats))
    f.close()
