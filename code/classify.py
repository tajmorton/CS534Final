#!/usr/bin/python
## Pick classifier & filter
import sys
import Orange, orange, orngTest, orngStat
import proj_utils
import os

"""
Trains given classifier on the given dataset.
"""
def train_classifier(data, type, filter):
	if type == "tree" or type == "c4.5" or type == "decision_tree":
		learner = orange.C45Learner()
	elif type == "bayes" or type == "naive" or type == "naive_bayes":
		learner = orange.BayesLearner()
	elif type == "svm" or type == "linear_svm":
		learner = Orange.classification.svm.LinearSVMLearner()
	elif type == "logreg" or type == "regression":
		learner = Orange.classification.logreg.LogRegLearner()
	else:
		print "Invalid Learner Type\n"
		exit()
		
	if filter == 0:
		classifier = learner(data)
	else:
		filtered_learner = Orange.feature.selection.FilteredLearner(learner, filter=Orange.feature.selection.FilterBestN(n=filter), name='filtered')
		classifier = filtered_learner(data)
		
	return classifier

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print >> sys.stderr, "Usage: classify.py [TAB_FILE] [classifier: tree, bayes, svm, logreg] [number to filter, 0 -> no filtering]"
        sys.exit(1)

    data = proj_utils.load_data(sys.argv[1])
    type = sys.argv[2]
    features = int(sys.argv[3])

    train_data, test_data = proj_utils.partition_data(data)

    model = train_classifier(train_data, type, features)
    train_CA, train_results = proj_utils.test_classifier(model, train_data)
    test_CA, test_results = proj_utils.test_classifier(model, test_data)

    #print "Train Accuracy: %f, Test Accuracy: %f" % (train_CA, test_CA)
    train_stats = proj_utils.get_stats(train_results)
    test_stats = proj_utils.get_stats(test_results)

    print "Train:\n%s" % str(train_stats)
    print "\nTest:\n%s" % str(test_stats)
	
    if features != 0:
		print "Features selected:\n%s" % str(model.domain)
		filename = os.path.dirname(__file__) + '\\results\\' + type + '_filtered_' + str(features) + '.txt'
    else:
		filename = os.path.dirname(__file__) + '\\results\\' + type + '.txt'
		
    f = open(filename, 'w+')
    f.write("Train:\n")
    f.write(str(train_stats) + "\n")
    f.write("Test:\n")
    f.write(str(test_stats))
    if features != 0:
		f.write("\nFeatures Chosen:\n")
		f.write(str(model.domain))
    f.close()