#!/usr/bin/python
## Naive Bayes Classifier
import sys
import Orange, orange, orngTest, orngStat
import proj_utils

"""
Trains a Naive Bayes classifier on the given dataset.
"""
def train_classifier(data):
    classifier = orange.BayesLearner(data)
    return classifier

"""
Computes training accuracy of a model over the given
dataset. Returns a tuple containing the classification
accuracy and the ExperimentResults object for further
tests (is desired).
"""
def test_classifier(model, data):
    res = orngTest.testOnData( (model,), data) # testOnData requires a list of models, so convert model into a tuple of length 1

    class_accuracy = orngStat.CA(res)[0]
    return class_accuracy, res

def get_confusion_matrix(results, cutoff = 0.5):
    cms = orngStat.confusionMatrices(results, cutoff = cutoff)

    return cm[0]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, "Usage: naive_bayes.py [TAB_FILE]"
        sys.exit(1)

    data = proj_utils.load_data(sys.argv[1])

    train_data, test_data = proj_utils.partition_data(data)

    model = train_classifier(train_data)
    train_CA, train_results = test_classifier(model, train_data)
    test_CA, test_results = test_classifier(model, test_data)

    print "Train Accuracy: %f, Test Accuracy: %f" % (train_CA, test_CA)
