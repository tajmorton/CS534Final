# Helper functions for dealing with data
import orange
import Orange

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

