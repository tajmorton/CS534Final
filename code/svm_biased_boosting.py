#!/usr/bin/python
## SVM Classifier
import sys
import Orange, orange, orngTest, orngStat
import proj_utils
import os

"""
Trains a Linear SVM classifier on the given dataset.
"""
def train_classifier(data):
    learner = Orange.classification.svm.LinearSVMLearner()
    classifier = learner(data)
    return classifier

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print >> sys.stderr, "Usage: naive_bayes.py [TAB_FILE] [CLASS_PROPORTIONS]"
        sys.exit(1)

    proportions = 0.1
    if len(sys.argv) >= 3:
        proportions = float(sys.argv[2])

    data = proj_utils.load_data(sys.argv[1])

    train_data, test_data = proj_utils.partition_data(data)

    resampled_train = proj_utils.oversample_class(train_data, 'ad', proportions)

    model = train_classifier(resampled_train)
    train_CA, train_results = proj_utils.test_classifier(model, resampled_train)
    test_CA, test_results = proj_utils.test_classifier(model, test_data)

    #print "Train Accuracy: %f, Test Accuracy: %f" % (train_CA, test_CA)
    train_stats = proj_utils.get_stats(train_results)
    test_stats = proj_utils.get_stats(test_results)

    print "Train:\n%s" % str(train_stats)
    print "\nTest:\n%s\n" % str(test_stats)
