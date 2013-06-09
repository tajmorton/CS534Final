#!/usr/bin/python
import sys
import os
import ast

all_results = {}
for filename in sys.argv[1:]:
    fname_base = os.path.basename(filename)
    class_type, is_filtered, count = fname_base.split('.')[0].split('_')

    if count not in all_results:
        all_results[count] = {}

    if class_type not in all_results[count]:
        all_results[count][class_type] = {}

    with open(filename, 'r') as f:
        f.readline() # skip the first 3 lines
        f.readline()
        f.readline()
        test_perf = f.readline() # read the test results line

        perf = ast.literal_eval(test_perf)
        all_results[count][class_type] = perf

for count, class_types in all_results.iteritems():
    print '"Num Features",',
    for class_type in class_types.keys():
        print '"%s Sensitvitiy", "%s Specificity", ' % (class_type, class_type),

    break

print

for count, class_types in all_results.iteritems():
    print "%s, " % count,
    for class_type, results in class_types.iteritems():
        print "%.4f, %.4f, " % (results['Sensitivity'], results['Specificity']),

    print ""
