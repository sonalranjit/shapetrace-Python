__author__ = 'sonal'

import numpy as np

poseTxt = "data/May 12/pose-461.535240001.txt"

def parsePoseQuat(file):
    x = []
    y = []
    z = []
    q1 = []
    q2 = []
    q3 = []
    q4 = []
    with open(file) as f:
        for _ in xrange(2):
            next(f)
        for line in (f):
            element = line.strip().split(",")
            x = float(element[3][1:len(element[3])])
            y = float(element[4])
            z = float(element[5][0:(len(element[5])-1)])
            q1 = float(element[6][2:len(element[6])])
            q2 = float(element[7])
            q3 = float(element[8])
            q4 = float(element[9][0:(len(element[9])-1)])

    XYZ = np.array(np.hstack(((x,y),z))).reshape((3,1))
    quat12 = np.hstack((q1,q2))
    quat34 = np.hstack((q3,q4))
    QT = np.array(np.hstack((quat12,quat34))).reshape(4,1)

    print XYZ

parsePoseQuat(poseTxt)