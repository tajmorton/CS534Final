#!/usr/bin/python
import sys
from collections import OrderedDict

def parse_names(names_filename):
    classes = []
    feature_list = OrderedDict()

    for line in open(names_filename, 'r'):
        line = line.split('|', 1)[0]
        if len(line.strip()) == 0:
            continue

        if line.strip()[-1] != '.':
            classes = map(lambda x: x.strip(), line.split(','))
            continue

        # a feature line
        name, f_type = line.split(': ')
        if f_type.startswith("continuous"):
            feature_list[name] = "continuous"
        else: # everything is is discrete (binary) in this data
            feature_list[name] = "discrete"

    return feature_list, classes

def write_tab_header(tab_out, feature_list, class_name):
    for f in feature_list.keys(): 
        tab_out.write("%s\t" % f)

    tab_out.write("%s\n" % class_name)

    for v in feature_list.values():
        tab_out.write("%s\t" % v)

    tab_out.write("discrete\n") # class label is discrete

    tab_out.write("\t"*len(feature_list)) # empty fields for each feature
    tab_out.write("class\n") # mark last class as `class` for Orange

def write_data_tab(tab_out, datafile):
    for line in open(datafile):
        tab_out.write(line.strip()[:-1].replace(",","\t"))
        tab_out.write("\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print >> sys.stderr, "Usage: process_to_tab.py [ad.names] [ad.data] [TAB_OUT]"
        print >> sys.stderr, "Converts dataset from C4.5 format to TAB Orange format."
        sys.exit(1)

    features, classes = parse_names(sys.argv[1])

    with open(sys.argv[3], 'w') as tab_out:
        write_tab_header(tab_out, features, "image_class")
        write_data_tab(tab_out, sys.argv[2])
