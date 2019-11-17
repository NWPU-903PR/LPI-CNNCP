# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:52:10 2018

@author: XIXI
"""

import numpy
import numpy as np
from keras.utils.np_utils import to_categorical
from functools import reduce
from scipy.io import loadmat


def lncstr2int(inputvalue):

    s = numpy.zeros((4, len(inputvalue)), 'float32')

    for j in range(len(inputvalue)):
        if inputvalue[j] == 'A':
            s[:, j] = [1, 0, 0, 0]
        elif inputvalue[j] == 'U' :
            s[:, j] = [0, 1, 0, 0]
        elif inputvalue[j] == 'C' :
            s[:, j] = [0, 0, 1, 0]
        elif inputvalue[j] == 'G':
            s[:, j] = [0, 0, 0, 1]
        else:
#            s[:, j] = [0.25, 0.25, 0.25, 0.25]
            s[:, j] = [0, 0, 0, 0]


    return s

def RNA_2(inputvalue):

    s = np.zeros((16,len(inputvalue)-1 ), 'float32')

    for j in range(len(inputvalue)-1):
        if inputvalue[j] == 'G'and inputvalue[j+1]=='G':
            s[:, j] = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='A':
            s[:, j] = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='C':
            s[:, j] = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='U':
            s[:, j] = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
            
        elif inputvalue[j] == 'A'and inputvalue[j+1]=='G':
            s[:, j] = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'A'and inputvalue[j+1]=='A':
            s[:, j] = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'A'and inputvalue[j+1]=='C':
            s[:, j] = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'A'and inputvalue[j+1]=='U':
            s[:, j] = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
            
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='G':
            s[:, j] = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='A':
            s[:, j] = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='C':
            s[:, j] = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='U':
            s[:, j] = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
            
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='G':
            s[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='A':
            s[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='C':
            s[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='U':
            s[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        else:
            s[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    return s


def RNA_3(inputvalue):

    ss = np.zeros((64,len(inputvalue)-2 ), 'float32')

    for j in range(len(inputvalue)-2):
        if inputvalue[j] == 'A'and inputvalue[j+1]=='A'and inputvalue[j+2]=='A':
            ss[:, j] = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='A'and inputvalue[j+2]=='A':
            ss[:, j] = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='A'and inputvalue[j+2]=='A':
            ss[:, j] = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='A'and inputvalue[j+2]=='A':
            ss[:, j] = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'A'and inputvalue[j+1]=='C'and inputvalue[j+2]=='A':
            ss[:, j] = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='C'and inputvalue[j+2]=='A':
            ss[:, j] = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='C'and inputvalue[j+2]=='A':
            ss[:, j] = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='C'and inputvalue[j+2]=='A':
            ss[:, j] = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'A'and inputvalue[j+1]=='G'and inputvalue[j+2]=='A':
            ss[:, j] = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='G'and inputvalue[j+2]=='A':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='G'and inputvalue[j+2]=='A':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='G'and inputvalue[j+2]=='A':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'A'and inputvalue[j+1]=='U'and inputvalue[j+2]=='A':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='U'and inputvalue[j+2]=='A':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='U'and inputvalue[j+2]=='A':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='U'and inputvalue[j+2]=='A':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        elif inputvalue[j] == 'A'and inputvalue[j+1]=='A'and inputvalue[j+2]=='C':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='A'and inputvalue[j+2]=='C':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='A'and inputvalue[j+2]=='C':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='A'and inputvalue[j+2]=='C':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'A'and inputvalue[j+1]=='C'and inputvalue[j+2]=='C':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='C'and inputvalue[j+2]=='C':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='C'and inputvalue[j+2]=='C':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='C'and inputvalue[j+2]=='C':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'A'and inputvalue[j+1]=='G'and inputvalue[j+2]=='C':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='G'and inputvalue[j+2]=='C':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='G'and inputvalue[j+2]=='C':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='G'and inputvalue[j+2]=='C':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'A'and inputvalue[j+1]=='U'and inputvalue[j+2]=='C':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='U'and inputvalue[j+2]=='C':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='U'and inputvalue[j+2]=='C':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='U'and inputvalue[j+2]=='C':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        elif inputvalue[j] == 'A'and inputvalue[j+1]=='A'and inputvalue[j+2]=='G':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='A'and inputvalue[j+2]=='G':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='A'and inputvalue[j+2]=='G':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='A'and inputvalue[j+2]=='G':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'A'and inputvalue[j+1]=='C'and inputvalue[j+2]=='G':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='C'and inputvalue[j+2]=='G':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='C'and inputvalue[j+2]=='G':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='C'and inputvalue[j+2]=='G':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'A'and inputvalue[j+1]=='G'and inputvalue[j+2]=='G':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='G'and inputvalue[j+2]=='G':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='G'and inputvalue[j+2]=='G':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='G'and inputvalue[j+2]=='G':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'A'and inputvalue[j+1]=='U'and inputvalue[j+2]=='G':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='U'and inputvalue[j+2]=='G':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='U'and inputvalue[j+2]=='G':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='U'and inputvalue[j+2]=='G':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        elif inputvalue[j] == 'A'and inputvalue[j+1]=='A'and inputvalue[j+2]=='U':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='A'and inputvalue[j+2]=='U':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='A'and inputvalue[j+2]=='U':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='A'and inputvalue[j+2]=='U':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'A'and inputvalue[j+1]=='C'and inputvalue[j+2]=='U':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='C'and inputvalue[j+2]=='U':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='C'and inputvalue[j+2]=='U':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='C'and inputvalue[j+2]=='U':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'A'and inputvalue[j+1]=='G'and inputvalue[j+2]=='U':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='G'and inputvalue[j+2]=='U':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='G'and inputvalue[j+2]=='U':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='G'and inputvalue[j+2]=='U':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        elif inputvalue[j] == 'A'and inputvalue[j+1]=='U'and inputvalue[j+2]=='U':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
        elif inputvalue[j] == 'C'and inputvalue[j+1]=='U'and inputvalue[j+2]=='U':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        elif inputvalue[j] == 'G'and inputvalue[j+1]=='U'and inputvalue[j+2]=='U':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        elif inputvalue[j] == 'U'and inputvalue[j+1]=='U'and inputvalue[j+2]=='U':
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        else:
            ss[:, j] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    return ss

def get_4_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**4
    for i in range(0,end):
        n=i
        ch0=chars[int(n%base)]
        n=n/base
        ch1=chars[int(n%base)]
        n=n/base
        ch2=chars[int(n%base)]
        n=n/base
        ch3=chars[int(n%base)]
        nucle_com.append(ch0 + ch1 + ch2 + ch3)
    return  nucle_com

def RNA_4(inputvalue):
    all_kmer=get_4_trids()
    ss = np.zeros((256,len(inputvalue)-3 ), 'float32')
    k = len(all_kmer[0])          
    for x in range(len(inputvalue) + 1- k):
        kmer = inputvalue[x:x+k]   
        if kmer in all_kmer:
            ind = all_kmer.index(kmer)
            ss[ind][x]=1
    return ss

def prostr2int(sequence_pro):
    
    c = numpy.zeros((len(sequence_pro), 7), 'float32')
    
    j=0
    while j<len(sequence_pro):
        if sequence_pro[j]=='1':
            c[j:j+1,:]=[1,0,0,0,0,0,0]
        elif sequence_pro[j]=='2':
            c[j:j+1,:]=[0,1,0,0,0,0,0]
        elif sequence_pro[j]=='3':
            c[j:j+1,:]=[0,0,1,0,0,0,0]
        elif sequence_pro[j]=='4':
            c[j:j+1,:]=[0,0,0,1,0,0,0]
        elif sequence_pro[j]=='5':
            c[j:j+1,:]=[0,0,0,0,1,0,0]
        elif sequence_pro[j]=='6':
            c[j:j+1,:]=[0,0,0,0,0,1,0]
        elif sequence_pro[j]=='7':
            c[j:j+1,:]=[0,0,0,0,0,0,1]
        else :
            c[j:j+1,:]=[0,0,0,0,0,0,0]
        j=j+1

    d=c.transpose()

    return d

def protein_2(sequence_pro):
    
    c = numpy.zeros((len(sequence_pro)-1, 49), 'float32')
    
    j=0
    while j<(len(sequence_pro)-1):
        if sequence_pro[j]=='1' and sequence_pro[j+1]=='1':
            c[j:j+1,:]=[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='1' and sequence_pro[j+1]=='2':
            c[j:j+1,:]=[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='1' and sequence_pro[j+1]=='3':
            c[j:j+1,:]=[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='1' and sequence_pro[j+1]=='4':
            c[j:j+1,:]=[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='1' and sequence_pro[j+1]=='5':
            c[j:j+1,:]=[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='1' and sequence_pro[j+1]=='6':
            c[j:j+1,:]=[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='1' and sequence_pro[j+1]=='7':
            c[j:j+1,:]=[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        elif sequence_pro[j]=='2' and sequence_pro[j+1]=='1':
            c[j:j+1,:]=[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='2' and sequence_pro[j+1]=='2':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='2' and sequence_pro[j+1]=='3':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='2' and sequence_pro[j+1]=='4':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='2' and sequence_pro[j+1]=='5':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='2' and sequence_pro[j+1]=='6':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='2' and sequence_pro[j+1]=='7':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        elif sequence_pro[j]=='3' and sequence_pro[j+1]=='1':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='3' and sequence_pro[j+1]=='2':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='3' and sequence_pro[j+1]=='3':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='3' and sequence_pro[j+1]=='4':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='3' and sequence_pro[j+1]=='5':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='3' and sequence_pro[j+1]=='6':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='3' and sequence_pro[j+1]=='7':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


        elif sequence_pro[j]=='4' and sequence_pro[j+1]=='1':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='4' and sequence_pro[j+1]=='2':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='4' and sequence_pro[j+1]=='3':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='4' and sequence_pro[j+1]=='4':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='4' and sequence_pro[j+1]=='5':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='4' and sequence_pro[j+1]=='6':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='4' and sequence_pro[j+1]=='7':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        elif sequence_pro[j]=='5' and sequence_pro[j+1]=='1':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='5' and sequence_pro[j+1]=='2':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='5' and sequence_pro[j+1]=='3':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='5' and sequence_pro[j+1]=='4':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='5' and sequence_pro[j+1]=='5':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='5' and sequence_pro[j+1]=='6':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='5' and sequence_pro[j+1]=='7':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        elif sequence_pro[j]=='6' and sequence_pro[j+1]=='1':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='6' and sequence_pro[j+1]=='2':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='6' and sequence_pro[j+1]=='3':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='6' and sequence_pro[j+1]=='4':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='6' and sequence_pro[j+1]=='5':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='6' and sequence_pro[j+1]=='6':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
        elif sequence_pro[j]=='6' and sequence_pro[j+1]=='7':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]

        elif sequence_pro[j]=='7' and sequence_pro[j+1]=='1':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        elif sequence_pro[j]=='7' and sequence_pro[j+1]=='2':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        elif sequence_pro[j]=='7' and sequence_pro[j+1]=='3':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        elif sequence_pro[j]=='7' and sequence_pro[j+1]=='4':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
        elif sequence_pro[j]=='7' and sequence_pro[j+1]=='5':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        elif sequence_pro[j]=='7' and sequence_pro[j+1]=='6':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        elif sequence_pro[j]=='7' and sequence_pro[j+1]=='7':
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
       
        else :
            c[j:j+1,:]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        j=j+1

    d=c.transpose()

    return d

def get_3_protein_trids():
    nucle_com = []
    chars = ['1', '2', '3', '4', '5', '6', '7']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[int(n%base)]
        n=n/base
        ch1=chars[int(n%base)]
        n=n/base
        ch2=chars[int(n%base)]
        nucle_com.append(ch0 + ch1 + ch2)
    return  nucle_com

def protein_3(sequence_pro):
    c = numpy.zeros((343,len(sequence_pro)-2), 'float32')
    all_kmer=get_3_protein_trids()
    k = len(all_kmer[0])          
    for x in range(len(sequence_pro) + 1- k):
        kmer = sequence_pro[x:x+k]   
        if kmer in all_kmer:
            ind = all_kmer.index(kmer)
            c[ind][x]=1
    return c

def replace_char(s, oldChar, newChar ):
    return reduce(lambda s, char: s.replace(char, newChar), oldChar, s)
    
def translate2int(inputvalue):
    
    value_label = inputvalue.split('$')
    label = eval(value_label[0])
    value = value_label[3].split('#')
    value1 = value[1][0:-1].upper()
    value2 = value[0].upper()
    value2=replace_char(value2, 'AGV', '1')
    value2=replace_char(value2, 'ILFP', '2')
    value2=replace_char(value2, 'YMTS', '3')
    value2=replace_char(value2, 'HNQW', '4')
    value2=replace_char(value2, 'RK', '5')
    value2=replace_char(value2, 'DE', '6')
    value2=replace_char(value2, 'C', '7')
    return value1, value2, label

def zeros_padding_mothod(filename,len_pro,len_lnc):
    zeros_padding=[]
    with open(filename) as f1:
        data1=f1.readlines()
        for line in data1:
            data2=line.split('$')
            data3=data2[3]
            data4=data3.split('#')
            l1=len(data4[0])
            l2=len(data4[1][:-1])
            i=len_pro-l1
            k=len_lnc-l2
            pro_seq=data4[0]+i*'X'
            data5=data4[1][:-1]
            lnc_seq=data5+k*'X'
            zeros_padding.append(data2[0]+'$'+data2[1]+'$'+data2[2]+'$'+pro_seq+'#'+lnc_seq+'\n')
    return  zeros_padding

def copy_crop_method(filename,len_pro,len_lnc):
    copy_crop=[]
    with open(filename) as f1:
        data1 = f1.readlines()
        for line in data1:
            data2=line.split('$')
            data3=data2[3]
            data4=data3.split('#')
            l1=len(data4[0])
            l2=len(data4[1][:-1])
            i=len_pro//l1
            j=len_pro%l1
            k=len_lnc//l2
            l=len_lnc%l2
            pro_seq=data4[0]*i+data4[0][:j]
            data5=data4[1][:-1]
            lnc_seq=data5*k+data5[:l]
            copy_crop.append(data2[0]+'$'+data2[1]+'$'+data2[2]+'$'+pro_seq+'#'+lnc_seq+'\n')
    return  copy_crop

def data_one_one_preprocess(train_filename,len_pro,len_lnc,copy):

#Calculate the sequence length of lncRNA and protein
    LEN_peo=len_pro
    LEN_lnc=len_lnc

#Processing sequence
    if copy:
       fa11=copy_crop_method(train_filename,len_pro,len_lnc)
    else:
       fa11 = zeros_padding_mothod(train_filename, len_pro, len_lnc)
    train_x = []
    train_y = []
    label_train=[]
    for i in range(len(fa11)):
        train_xx,train_yy,label_train1=translate2int(fa11[i])
        train_x.append(train_xx)
        train_y.append(train_yy)
        label_train.append(label_train1)
    feature_lnc_train=[]
    feature_pro_train=[]
    feature_lnc_train1=np.zeros((len(fa11),4,LEN_lnc))
    feature_pro_train1=np.zeros((len(fa11),7,LEN_peo))
    #label_train = np.zeros((len(fa11),))


    for i in range(len(fa11)):
# =============================================================================
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
# =============================================================================
        train_x[i]=train_x[i].center(len_lnc,'X')
        train_y[i]=train_y[i].center(len_pro,'X')
        feature_lnc_train.append(lncstr2int(train_x[i]))
        feature_pro_train.append(prostr2int(train_y[i]))
        label_train[i]=int(label_train[i])
    for i in range(len(fa11)):
        feature_lnc_train1[i,:]=feature_lnc_train[i]
        feature_pro_train1[i,:]=feature_pro_train[i]
    feature_lnc_train1=feature_lnc_train1.reshape(len(fa11),4,LEN_lnc,1)
    feature_pro_train1=feature_pro_train1.reshape(len(fa11),7,LEN_peo,1)
    label_train=to_categorical(label_train, num_classes=2)


    return feature_pro_train1,feature_lnc_train1,label_train

def data_one_two_preprocess(train_filename,len_pro,len_lnc,copy):


    LEN_peo=len_pro
    LEN_lnc=len_lnc-1

    if copy:
       fa11=copy_crop_method(train_filename,len_pro,len_lnc)
    else:
       fa11 = zeros_padding_mothod(train_filename, len_pro, len_lnc)
    train_x = []
    train_y = []
    label_train=[]
    for i in range(len(fa11)):
        train_xx,train_yy,label_train1=translate2int(fa11[i])
        train_x.append(train_xx)
        train_y.append(train_yy)
        label_train.append(label_train1)
    feature_lnc_train=[]
    feature_pro_train=[]
    feature_lnc_train1=np.zeros((len(fa11),16,LEN_lnc))
    feature_pro_train1=np.zeros((len(fa11),7,LEN_peo))
    #label_train = np.zeros((len(fa11),))


    for i in range(len(fa11)):
# =============================================================================
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
# =============================================================================
        train_x[i]=train_x[i].center(len_lnc,'X')
        train_y[i]=train_y[i].center(len_pro,'X')
        feature_lnc_train.append(RNA_2(train_x[i]))
        feature_pro_train.append(prostr2int(train_y[i]))
        label_train[i]=int(label_train[i])
    for i in range(len(fa11)):
        feature_lnc_train1[i,:]=feature_lnc_train[i]
        feature_pro_train1[i,:]=feature_pro_train[i]
    feature_lnc_train1=feature_lnc_train1.reshape(len(fa11),16,LEN_lnc,1)
    feature_pro_train1=feature_pro_train1.reshape(len(fa11),7,LEN_peo,1)
    label_train=to_categorical(label_train, num_classes=2)


    return feature_pro_train1,feature_lnc_train1,label_train


def data_one_three_preprocess(train_filename,len_pro,len_lnc,copy):


    LEN_peo=len_pro
    LEN_lnc=len_lnc-2

    if copy:
       fa11=copy_crop_method(train_filename,len_pro,len_lnc)
    else:
       fa11 = zeros_padding_mothod(train_filename, len_pro, len_lnc)
    train_x = []
    train_y = []
    label_train=[]
    for i in range(len(fa11)):
        train_xx,train_yy,label_train1=translate2int(fa11[i])
        train_x.append(train_xx)
        train_y.append(train_yy)
        label_train.append(label_train1)
    feature_lnc_train=[]
    feature_pro_train=[]
    feature_lnc_train1=np.zeros((len(fa11),64,LEN_lnc))
    feature_pro_train1=np.zeros((len(fa11),7,LEN_peo))
    #label_train = np.zeros((len(fa11),))


    for i in range(len(fa11)):
# =============================================================================
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
# =============================================================================
        train_x[i]=train_x[i].center(len_lnc,'X')
        train_y[i]=train_y[i].center(len_pro,'X')
        feature_lnc_train.append(RNA_3(train_x[i]))
        feature_pro_train.append(prostr2int(train_y[i]))
        label_train[i]=int(label_train[i])
    for i in range(len(fa11)):
        feature_lnc_train1[i,:]=feature_lnc_train[i]
        feature_pro_train1[i,:]=feature_pro_train[i]
    feature_lnc_train1=feature_lnc_train1.reshape(len(fa11),64,LEN_lnc,1)
    feature_pro_train1=feature_pro_train1.reshape(len(fa11),7,LEN_peo,1)
    label_train=to_categorical(label_train, num_classes=2)


    return feature_pro_train1,feature_lnc_train1,label_train


def data_one_four_preprocess(train_filename,len_pro,len_lnc,copy):

    LEN_peo=len_pro
    LEN_lnc=len_lnc-3

    if copy:
       fa11=copy_crop_method(train_filename,len_pro,len_lnc)
    else:
       fa11 = zeros_padding_mothod(train_filename, len_pro, len_lnc)
    train_x = []
    train_y = []
    label_train=[]
    for i in range(len(fa11)):
        train_xx,train_yy,label_train1=translate2int(fa11[i])
        train_x.append(train_xx)
        train_y.append(train_yy)
        label_train.append(label_train1)
    feature_lnc_train=[]
    feature_pro_train=[]
    feature_lnc_train1=np.zeros((len(fa11),256,LEN_lnc))
    feature_pro_train1=np.zeros((len(fa11),7,LEN_peo))
    #label_train = np.zeros((len(fa11),))


    for i in range(len(fa11)):
# =============================================================================
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
# =============================================================================
        train_x[i]=train_x[i].center(len_lnc,'X')
        train_y[i]=train_y[i].center(len_pro,'X')
        feature_lnc_train.append(RNA_4(train_x[i]))
        feature_pro_train.append(prostr2int(train_y[i]))
        label_train[i]=int(label_train[i])
    for i in range(len(fa11)):
        feature_lnc_train1[i,:]=feature_lnc_train[i]
        feature_pro_train1[i,:]=feature_pro_train[i]
    feature_lnc_train1=feature_lnc_train1.reshape(len(fa11),256,LEN_lnc,1)
    feature_pro_train1=feature_pro_train1.reshape(len(fa11),7,LEN_peo,1)
    label_train=to_categorical(label_train, num_classes=2)


    return feature_pro_train1,feature_lnc_train1,label_train

def data_two_one_preprocess(train_filename,len_pro,len_lnc,copy):


    LEN_peo=len_pro-1
    LEN_lnc=len_lnc

    if copy:
       fa11=copy_crop_method(train_filename,len_pro,len_lnc)
    else:
       fa11 = zeros_padding_mothod(train_filename, len_pro, len_lnc)
    train_x = []
    train_y = []
    label_train=[]
    for i in range(len(fa11)):
        train_xx,train_yy,label_train1=translate2int(fa11[i])
        train_x.append(train_xx)
        train_y.append(train_yy)
        label_train.append(label_train1)
    feature_lnc_train=[]
    feature_pro_train=[]
    feature_lnc_train1=np.zeros((len(fa11),4,LEN_lnc))
    feature_pro_train1=np.zeros((len(fa11),49,LEN_peo))
    #label_train = np.zeros((len(fa11),))


    for i in range(len(fa11)):
# =============================================================================
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
# =============================================================================
        train_x[i]=train_x[i].center(len_lnc,'X')
        train_y[i]=train_y[i].center(len_pro,'X')
        feature_lnc_train.append(lncstr2int(train_x[i]))
        feature_pro_train.append(protein_2(train_y[i]))
        label_train[i]=int(label_train[i])
    for i in range(len(fa11)):
        feature_lnc_train1[i,:]=feature_lnc_train[i]
        feature_pro_train1[i,:]=feature_pro_train[i]
    feature_lnc_train1=feature_lnc_train1.reshape(len(fa11),4,LEN_lnc,1)
    feature_pro_train1=feature_pro_train1.reshape(len(fa11),49,LEN_peo,1)
    label_train=to_categorical(label_train, num_classes=2)


    return feature_pro_train1,feature_lnc_train1,label_train

def data_two_two_preprocess(train_filename,len_pro,len_lnc,copy):


    LEN_peo=len_pro-1
    LEN_lnc=len_lnc-1

    if copy:
       fa11=copy_crop_method(train_filename,len_pro,len_lnc)
    else:
       fa11 = zeros_padding_mothod(train_filename, len_pro, len_lnc)
    train_x = []
    train_y = []
    label_train=[]
    for i in range(len(fa11)):
        train_xx,train_yy,label_train1=translate2int(fa11[i])
        train_x.append(train_xx)
        train_y.append(train_yy)
        label_train.append(label_train1)
    feature_lnc_train=[]
    feature_pro_train=[]
    feature_lnc_train1=np.zeros((len(fa11),16,LEN_lnc))
    feature_pro_train1=np.zeros((len(fa11),49,LEN_peo))
    #label_train = np.zeros((len(fa11),))


    for i in range(len(fa11)):
# =============================================================================
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
# =============================================================================
        train_x[i]=train_x[i].center(len_lnc,'X')
        train_y[i]=train_y[i].center(len_pro,'X')
        feature_lnc_train.append(RNA_2(train_x[i]))
        feature_pro_train.append(protein_2(train_y[i]))
        label_train[i]=int(label_train[i])
    for i in range(len(fa11)):
        feature_lnc_train1[i,:]=feature_lnc_train[i]
        feature_pro_train1[i,:]=feature_pro_train[i]
    feature_lnc_train1=feature_lnc_train1.reshape(len(fa11),16,LEN_lnc,1)
    feature_pro_train1=feature_pro_train1.reshape(len(fa11),49,LEN_peo,1)
    label_train=to_categorical(label_train, num_classes=2)


    return feature_pro_train1,feature_lnc_train1,label_train


def data_two_three_preprocess(train_filename,len_pro,len_lnc,copy):


    LEN_peo=len_pro-1
    LEN_lnc=len_lnc-2

    if copy:
       fa11=copy_crop_method(train_filename,len_pro,len_lnc)
    else:
       fa11 = zeros_padding_mothod(train_filename, len_pro, len_lnc)
    train_x = []
    train_y = []
    label_train=[]
    for i in range(len(fa11)):
        train_xx,train_yy,label_train1=translate2int(fa11[i])
        train_x.append(train_xx)
        train_y.append(train_yy)
        label_train.append(label_train1)
    feature_lnc_train=[]
    feature_pro_train=[]
    feature_lnc_train1=np.zeros((len(fa11),64,LEN_lnc))
    feature_pro_train1=np.zeros((len(fa11),49,LEN_peo))
    #label_train = np.zeros((len(fa11),))


    for i in range(len(fa11)):
# =============================================================================
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
# =============================================================================
        train_x[i]=train_x[i].center(len_lnc,'X')
        train_y[i]=train_y[i].center(len_pro,'X')
        feature_lnc_train.append(RNA_3(train_x[i]))
        feature_pro_train.append(protein_2(train_y[i]))
        label_train[i]=int(label_train[i])
    for i in range(len(fa11)):
        feature_lnc_train1[i,:]=feature_lnc_train[i]
        feature_pro_train1[i,:]=feature_pro_train[i]
    feature_lnc_train1=feature_lnc_train1.reshape(len(fa11),64,LEN_lnc,1)
    feature_pro_train1=feature_pro_train1.reshape(len(fa11),49,LEN_peo,1)
    label_train=to_categorical(label_train, num_classes=2)

    return feature_pro_train1,feature_lnc_train1,label_train

def data_two_four_preprocess(train_filename,len_pro,len_lnc,copy):


    LEN_peo=len_pro-1
    LEN_lnc=len_lnc-3

    if copy:
       fa11=copy_crop_method(train_filename,len_pro,len_lnc)
    else:
       fa11 = zeros_padding_mothod(train_filename, len_pro, len_lnc)
    train_x = []
    train_y = []
    label_train=[]
    for i in range(len(fa11)):
        train_xx,train_yy,label_train1=translate2int(fa11[i])
        train_x.append(train_xx)
        train_y.append(train_yy)
        label_train.append(label_train1)
    feature_lnc_train=[]
    feature_pro_train=[]
    feature_lnc_train1=np.zeros((len(fa11),256,LEN_lnc))
    feature_pro_train1=np.zeros((len(fa11),49,LEN_peo))
    #label_train = np.zeros((len(fa11),))


    for i in range(len(fa11)):
# =============================================================================
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
# =============================================================================
        train_x[i]=train_x[i].center(len_lnc,'X')
        train_y[i]=train_y[i].center(len_pro,'X')
        feature_lnc_train.append(RNA_4(train_x[i]))
        feature_pro_train.append(protein_2(train_y[i]))
        label_train[i]=int(label_train[i])
    for i in range(len(fa11)):
        feature_lnc_train1[i,:]=feature_lnc_train[i]
        feature_pro_train1[i,:]=feature_pro_train[i]
    feature_lnc_train1=feature_lnc_train1.reshape(len(fa11),256,LEN_lnc,1)
    feature_pro_train1=feature_pro_train1.reshape(len(fa11),49,LEN_peo,1)
    label_train=to_categorical(label_train, num_classes=2)

    return feature_pro_train1,feature_lnc_train1,label_train



def data_three_one_preprocess(train_filename,len_pro,len_lnc,copy):


    LEN_peo=len_pro-2
    LEN_lnc=len_lnc

    if copy:
       fa11=copy_crop_method(train_filename,len_pro,len_lnc)
    else:
       fa11 = zeros_padding_mothod(train_filename, len_pro, len_lnc)
    train_x = []
    train_y = []
    label_train=[]
    for i in range(len(fa11)):
        train_xx,train_yy,label_train1=translate2int(fa11[i])
        train_x.append(train_xx)
        train_y.append(train_yy)
        label_train.append(label_train1)
    feature_lnc_train=[]
    feature_pro_train=[]
    feature_lnc_train1=np.zeros((len(fa11),4,LEN_lnc))
    feature_pro_train1=np.zeros((len(fa11),343,LEN_peo))
    #label_train = np.zeros((len(fa11),))


    for i in range(len(fa11)):
# =============================================================================
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
# =============================================================================
        train_x[i]=train_x[i].center(len_lnc,'X')
        train_y[i]=train_y[i].center(len_pro,'X')
        feature_lnc_train.append(lncstr2int(train_x[i]))
        feature_pro_train.append(protein_3(train_y[i]))
        label_train[i]=int(label_train[i])
    for i in range(len(fa11)):
        feature_lnc_train1[i,:]=feature_lnc_train[i]
        feature_pro_train1[i,:]=feature_pro_train[i]
    feature_lnc_train1=feature_lnc_train1.reshape(len(fa11),4,LEN_lnc,1)
    feature_pro_train1=feature_pro_train1.reshape(len(fa11),343,LEN_peo,1)
    label_train=to_categorical(label_train, num_classes=2)


    return feature_pro_train1,feature_lnc_train1,label_train

def data_three_two_preprocess(train_filename,len_pro,len_lnc,copy):


    LEN_peo=len_pro-2
    LEN_lnc=len_lnc-1

    if copy:
       fa11=copy_crop_method(train_filename,len_pro,len_lnc)
    else:
       fa11 = zeros_padding_mothod(train_filename, len_pro, len_lnc)
    train_x = []
    train_y = []
    label_train=[]
    for i in range(len(fa11)):
        train_xx,train_yy,label_train1=translate2int(fa11[i])
        train_x.append(train_xx)
        train_y.append(train_yy)
        label_train.append(label_train1)
    feature_lnc_train=[]
    feature_pro_train=[]
    feature_lnc_train1=np.zeros((len(fa11),16,LEN_lnc))
    feature_pro_train1=np.zeros((len(fa11),343,LEN_peo))
    #label_train = np.zeros((len(fa11),))


    for i in range(len(fa11)):
# =============================================================================
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
# =============================================================================
        train_x[i]=train_x[i].center(len_lnc,'X')
        train_y[i]=train_y[i].center(len_pro,'X')
        feature_lnc_train.append(RNA_2(train_x[i]))
        feature_pro_train.append(protein_3(train_y[i]))
        label_train[i]=int(label_train[i])
    for i in range(len(fa11)):
        feature_lnc_train1[i,:]=feature_lnc_train[i]
        feature_pro_train1[i,:]=feature_pro_train[i]
    feature_lnc_train1=feature_lnc_train1.reshape(len(fa11),16,LEN_lnc,1)
    feature_pro_train1=feature_pro_train1.reshape(len(fa11),343,LEN_peo,1)
    label_train=to_categorical(label_train, num_classes=2)


    return feature_pro_train1,feature_lnc_train1,label_train


def data_three_three_preprocess(train_filename,len_pro,len_lnc,copy):


    LEN_peo=len_pro-2
    LEN_lnc=len_lnc-2

    if copy:
       fa11=copy_crop_method(train_filename,len_pro,len_lnc)
    else:
       fa11 = zeros_padding_mothod(train_filename, len_pro, len_lnc)
    train_x = []
    train_y = []
    label_train=[]
    for i in range(len(fa11)):
        train_xx,train_yy,label_train1=translate2int(fa11[i])
        train_x.append(train_xx)
        train_y.append(train_yy)
        
        label_train.append(label_train1)
    feature_lnc_train=[]
    feature_pro_train=[]
    feature_lnc_train1=np.zeros((len(fa11),64,LEN_lnc))
    feature_pro_train1=np.zeros((len(fa11),343,LEN_peo))
    #label_train = np.zeros((len(fa11),))


    for i in range(len(fa11)):
# =============================================================================
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
# =============================================================================
        train_x[i]=train_x[i].center(len_lnc,'X')
        train_y[i]=train_y[i].center(len_pro,'X')
        feature_lnc_train.append(RNA_3(train_x[i]))
        feature_pro_train.append(protein_3(train_y[i]))
        label_train[i]=int(label_train[i])
    for i in range(len(fa11)):
        feature_lnc_train1[i,:]=feature_lnc_train[i]
        feature_pro_train1[i,:]=feature_pro_train[i]
    feature_lnc_train1=feature_lnc_train1.reshape(len(fa11),64,LEN_lnc,1)
    feature_pro_train1=feature_pro_train1.reshape(len(fa11),343,LEN_peo,1)
    label_train=to_categorical(label_train, num_classes=2)

    return feature_pro_train1,feature_lnc_train1,label_train

def data_three_four_preprocess(train_filename,len_pro,len_lnc,copy):


    LEN_peo=len_pro-2
    LEN_lnc=len_lnc-3

    if copy:
       fa11=copy_crop_method(train_filename,len_pro,len_lnc)
    else:
       fa11 = zeros_padding_mothod(train_filename, len_pro, len_lnc)
    train_x = []
    train_y = []
    label_train=[]
    for i in range(len(fa11)):
        train_xx,train_yy,label_train1=translate2int(fa11[i])
        train_x.append(train_xx)
        train_y.append(train_yy)
        label_train.append(label_train1)
    feature_lnc_train=[]
    feature_pro_train=[]
    feature_lnc_train1=np.zeros((len(fa11),256,LEN_lnc))
    feature_pro_train1=np.zeros((len(fa11),343,LEN_peo))
    #label_train = np.zeros((len(fa11),))


    for i in range(len(fa11)):
# =============================================================================
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
#         train_x[i]=string.ljust(train_x[i],LEN_lnc,'0')
# =============================================================================
        train_x[i]=train_x[i].center(len_lnc,'X')
        train_y[i]=train_y[i].center(len_pro,'X')
        feature_lnc_train.append(RNA_4(train_x[i]))
        feature_pro_train.append(protein_3(train_y[i]))
        label_train[i]=int(label_train[i])
    for i in range(len(fa11)):
        feature_lnc_train1[i,:]=feature_lnc_train[i]
        feature_pro_train1[i,:]=feature_pro_train[i]
    feature_lnc_train1=feature_lnc_train1.reshape(len(fa11),256,LEN_lnc,1)
    feature_pro_train1=feature_pro_train1.reshape(len(fa11),343,LEN_peo,1)
    label_train=to_categorical(label_train, num_classes=2)

    return feature_pro_train1,feature_lnc_train1,label_train

# xixi1=data_one_one_preprocess('C:/Users/XIXI/Desktop/test.txt',22,19,True)
# xixi2=data_one_two_preprocess('C:/Users/XIXI/Desktop/test.txt',22,19,True)
# xixi3=data_one_three_preprocess('C:/Users/XIXI/Desktop/test.txt',22,19,True)
# xixi4=data_one_four_preprocess('C:/Users/XIXI/Desktop/test.txt',22,19,True)
#
# xixi5=data_two_one_preprocess('C:/Users/XIXI/Desktop/test.txt',22,19,True)
# xixi6=data_two_two_preprocess('C:/Users/XIXI/Desktop/test.txt',22,19,True)
# xixi7=data_two_three_preprocess('C:/Users/XIXI/Desktop/test.txt',22,19,True)
# xixi8=data_two_four_preprocess('C:/Users/XIXI/Desktop/test.txt',22,19,True)
#
# xixi9=data_three_one_preprocess('C:/Users/XIXI/Desktop/test.txt',22,19,True)
# xixi10=data_three_two_preprocess('C:/Users/XIXI/Desktop/test.txt',22,19,True)
# xixi11=data_three_three_preprocess('C:/Users/XIXI/Desktop/test.txt',22,19,True)
# xixi12=data_three_four_preprocess('C:/Users/XIXI/Desktop/test.txt',22,19,True)
