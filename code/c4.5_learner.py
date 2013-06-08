#!/usr/bin/python
## C4.5 Tree Classifier
import sys
import Orange, orange, orngTest, orngStat, orngEnsemble, orngTree
import proj_utils
import os

"""
Trains a decision tree classifier on the given dataset.
"""
def train_classifier(data):
    classifier = Orange.classification.tree.C45Learner(data)
    return classifier 

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print >> sys.stderr, "Usage: c4.5_learner.py [TAB_FILE] [PROPORTIONS]"
        sys.exit(1)

    proportions = 0.1
    if len(sys.argv) >= 3:
        proportions = float(sys.argv[2])

    data = proj_utils.load_data(sys.argv[1])

    train_data, test_data = proj_utils.partition_data(data)
    resampled_train_data = proj_utils.oversample_class(train_data, 'ad', proportions)

    model = train_classifier(resampled_train_data)
    train_CA, train_results = proj_utils.test_classifier(model, resampled_train_data)
    test_CA, test_results = proj_utils.test_classifier(model, test_data)

    #print "Train Accuracy: %f, Test Accuracy: %f" % (train_CA, test_CA)
    train_stats = proj_utils.get_stats(train_results)
    test_stats = proj_utils.get_stats(test_results)

    print "Train:\n%s" % str(train_stats)
    print "\nTest:\n%s" % str(test_stats)


