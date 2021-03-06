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

    print "\"Proportion\"",
    proj_utils.print_csv_header()
    #for prop in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    for prop in (1.0,1.5,2.0,2.5,3.0,3.5):
        resampled_train_data = proj_utils.undersample_class(train_data, 'nonad', prop)

        model = train_classifier(resampled_train_data)
        train_CA, train_results = proj_utils.test_classifier(model, resampled_train_data)
        test_CA, test_results = proj_utils.test_classifier(model, test_data)

        #print "Train Accuracy: %f, Test Accuracy: %f" % (train_CA, test_CA)
        #train_stats = proj_utils.get_stats(train_results)
        test_stats = proj_utils.get_stats(test_results)

        #print "Train:\n%s" % str(train_stats)
        #print "\nTest:\n%s\n" % str(test_stats)
        print "%f, " % prop,
        proj_utils.print_results_csv(test_stats)
