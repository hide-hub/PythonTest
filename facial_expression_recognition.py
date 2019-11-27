# facial expression recognition test
# facial data is provided by kaggle challenge project
# the code for recognition is for udemy lazyprogrammer's project

import csv
import numpy  as np
import pandas as pd
from matplotlib import pyplot as plt


##################################################
# definition of function
###################################################
def read_dataset( csv_file_name ):
    '''This function is specified to kaggle competition, Facial Expression Recognition.
    input is csv file name which contains facial expression data in particular format
    outputs are 3 arrays
    1st is image data
    2nd is expression (0:anger, 1:Happy)
    3rd is usage data (0:for training, 1:for public test, 2:for privat test)
    '''
    with open( csv_file_name ) as f:
        reader = csv.reader( f )
        l = [row for row in reader]

    # create empty lists
    X = []
    Y = []
    Usage = []
    l.pop(0) # drop first column
    for row in l:
        # separate the expression result (7 categories)
        Y.append( int(row[0]) )
        # extract image data
        X.append([int(val) for val in row[1].split()])
        # tag for training or test
        if   row[2] == 'Training':
            Usage.append( 0 )
        elif row[2] == 'PublicTest':
            Usage.append( 1 )
        else:
            Usage.append( 2 )

    # change list to array
    X = np.array(X)
    Y = np.array(Y)
    Usage = np.array(Usage)

    return X, Y, Usage

##################################################
# program body
##################################################




