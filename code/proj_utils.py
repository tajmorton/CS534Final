# Helper functions for dealing with data
import orange
import Orange
import orngTest
import orngStat

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
