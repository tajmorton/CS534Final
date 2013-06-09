# Helper functions for dealing with data
import orange
import Orange
import orngTest
import orngStat
import random

def load_data(filename):
    return Orange.data.Table(filename)

"""
Partitions data into train and test partitions.
Parameters:
    - data: An ExampleTable containing the data to split
    - percent_train: Percentage of data that is placed
      into the training partition. Any remaining examples
      are placed into the test partition:

Return:
    Tuple containing 2 ExampleTables. The first entry
    is the training data.
"""
def partition_data(data, percent_train = 0.5):
    indx = orange.MakeRandomIndices2(p0 = percent_train)
    train_indices = indx(data)

    train = data.select(train_indices)
    test = data.select(train_indices, negate=True)

    return (train, test)

def undersample_class(data, class_attr, proportion):
    new_data = Orange.data.Table(data.domain)

    matching_class = filter(lambda x: x.get_class() == class_attr, data)
    other_class = filter(lambda x: x.get_class() != class_attr, data)

    # want to figure out how many examples to sample so that
    # len(matching_class)/len(other_class) = proportion

    total_needed = proportion*len(other_class)
    num_to_take = int(round(total_needed - len(matching_class)))
    #print "Need to remove %d examples (%d total needed). Class size: %d %d" % (-num_to_take, total_needed, len(matching_class), len(other_class))
    
    if (num_to_take > 1):
        print "Can't oversample--returning original data."
        return data

    num_to_take = len(matching_class) + num_to_take
    sampled =  [random.choice(matching_class) for _ in xrange(num_to_take)]
    new_data.extend(sampled)
    new_data.extend(other_class)

    matching_class = filter(lambda x: x.get_class() == class_attr, new_data)
    other_class = filter(lambda x: x.get_class() != class_attr, new_data)
    #print "Resampled data: %d %d" % (len(matching_class), len(other_class))
    return new_data

def oversample_class(data, class_attr, proportion):
    new_data = Orange.data.Table(data)

    matching_class = filter(lambda x: x.get_class() == class_attr, data)
    other_class = filter(lambda x: x.get_class() != class_attr, data)

    # want to figure out how many examples to sample so that
    # len(matching_class)/len(other_class) = proportion

    total_needed = proportion*len(other_class)
    num_to_take = int(round(total_needed - len(matching_class)))
    #print "Need to take %d examples (%d total needed). Class size: %d %d" % (num_to_take, total_needed, len(matching_class), len(other_class))
    
    if (num_to_take < 1):
        print "Can't undersample--returning original data."
        return new_data

    sampled =  [random.choice(matching_class) for _ in xrange(num_to_take)]
    for s in sampled:
        new_data.append(s)

    matching_class = filter(lambda x: x.get_class() == class_attr, new_data)
    other_class = filter(lambda x: x.get_class() != class_attr, new_data)
    #print "Resampled data: %d %d" % (len(matching_class), len(other_class))
    return new_data

def get_stats(results):
    result_dict = {}
    cm = get_confusion_matrix(results)

    result_dict['Accuracy'] = orngStat.CA(results)[0]
    result_dict['Sensitivity'] = orngStat.sens(cm)
    result_dict['Specificity'] = orngStat.spec(cm)

    return result_dict

"""
Computes training accuracy of a model over the given
dataset. Returns a tuple containing the classification
accuracy and the ExperimentResults object for further
tests (if desired).
"""
def test_classifier(model, data):
    res = orngTest.testOnData( (model,), data) # testOnData requires a list of models, so convert model into a tuple of length 1

    class_accuracy = orngStat.CA(res)[0]
    return class_accuracy, res

def get_confusion_matrix(results, cutoff = 0.5):
    cms = orngStat.confusionMatrices(results, cutoff = cutoff)

    return cms[0]

def print_csv_header():
    print '"Accuracy", "Sensitivity", "Specificity"'

def print_results_csv(results):
    print "%.5f, %.5f, %.5f" % (results['Accuracy'], results['Sensitivity'], results['Specificity'])
