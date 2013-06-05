#!/usr/bin/python
## Feature Selector
import sys
import Orange, orange, orngTest, orngStat
import proj_utils
import os

"""
Trains Naive Bayes with selected features
"""
def train_classifier(data, features):
    lr = Orange.classification.logreg.LogRegLearner()
    learner = Orange.feature.selection.FilteredLearner(lr, filter=Orange.feature.selection.FilterBestN(n=features), name='filtered')
    lr_classifier = learner(data)
    return lr_classifier

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, "Usage: logistic_regression_filtered.py [TAB_FILE]"
        sys.exit(1)

    data = proj_utils.load_data(sys.argv[1])


    features = 10
    train_data, test_data = proj_utils.partition_data(data)
    


    model = train_classifier(train_data, features)
    train_CA, train_results = proj_utils.test_classifier(model, train_data)
    test_CA, test_results = proj_utils.test_classifier(model, test_data)

    #print "Train Accuracy: %f, Test Accuracy: %f" % (train_CA, test_CA)
    train_stats = proj_utils.get_stats(train_results)
    test_stats = proj_utils.get_stats(test_results)

    print "Train:\n%s" % str(train_stats)
    print "\nTest:\n%s" % str(test_stats)
    
    f = open(os.path.dirname(__file__) + '\\logisticRegressionFilteredResults' + str(features) + '.txt', 'w+')
    f.write("Train:\n")
    f.write(str(train_stats) + "\n")
    f.write("Test:\n")
    f.write(str(test_stats))
    f.close()
