__author__ = 'sonal'
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


depth1 = "data/April 29/depth-910.897600001.txt"
depth2 = "data/April 29/depth-911.094202001.txt"
depth3 = "data/April 29/depth-930.983771001.txt"

camXYZ1 = np.array([0.547282388237575, 0.157351187292134, -0.0254139935371636]).transpose()
camQuat1 = np.array([0.501725122125409, -0.240603849434341, -0.353766788261331, 0.751818295194313])
camXYZ2 = np.array([0.541529891261503,0.156750238065981,-0.0181067783655014]).transpose()
camQuat2 = np.array([0.513630923400241,-0.243933364155223,-0.348845268070899,0.744974340043651])
camXYZ3 = np.array([0.554824368194775,0.165557872552343,0.00639985921924491]).transpose()
camQuat3 = np.array([0.482640892361687,-0.26910679415059,-0.378285271369796,0.742657091682955])

def plot3d(xyz,col):

    fig = plt.figure(1)
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(xyz[0,:],xyz[1,:],xyz[2,:],s=5,c=col)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def parseDepth(file):
    x = []
    y = []
    z = []
    with open(file) as f:
        for _ in xrange(3):
            next(f)
        for line in (f):
            element = line.strip().split(",")
            for i in xrange(0,len(element),3):
                try:
                    array = element[i:i+3]
                    x.append(float(array[0]))
                    y.append(float(array[1]))
                    z.append(float(array[2]))
                except Exception:
                    pass

    x = np.array(x).reshape((len(x),1))
    y = np.array(y).reshape((len(y),1))
    z = np.array(z).reshape((len(z),1))

    XY = np.hstack((x,y))
    XYZ = np.hstack((XY,z))

    return XYZ


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
            print element
            x = float(element[3][1:len(element[3])])
            y = float(element[4])
            z = float(element[5][0:(len(element[5]) - 1)])
            q1 = float(element[6][2:len(element[6])])
            q2 = float(element[7])
            q3 = float(element[8])
            q4 = float(element[9][0:(len(element[9]) - 1)])
            print q1

    XYZ = np.hstack(((x, y), z))
    quat12 = np.hstack((q1, q2))
    quat34 = np.hstack((q3, q4))
    QT = np.hstack((quat12, quat34))

    return XYZ, QT

def quat2mat(qt):

    q11 = 1 - (2*qt[1]*qt[1]) - (2*qt[2]*qt[2])
    q12 = 2*qt[0]*qt[1] - 2*qt[2]*qt[3]
    q13 = 1*qt[0]*qt[2] + 2*qt[1]*qt[3]
    q21 = 2*qt[0]*qt[1] + 2*qt[2]*qt[3]
    q22 = 1 - 2*qt[0]*qt[0] - 2*qt[2]*qt[2]
    q23 = 2*qt[1]*qt[2] - 2*qt[0]*qt[3]
    q31 = 2*qt[0]*qt[2] - 2*qt[1]*qt[3]
    q32 = 2*qt[1]*qt[2] + 2*qt[0]*qt[3]
    q33 = 1 - 2*qt[0]*qt[0] - 2*qt[1]*qt[1]



    rotMat = np.array([[q11, q12, q13], [q21, q22, q23], [q31, q32, q33]])

    return rotMat

depths1 = parseDepth(depth1)
rotMat1 = quat2mat(camQuat1)
depths2 = parseDepth(depth2)
rotMat2 = quat2mat(camQuat2)
depths3 = parseDepth(depth3)
rotMat3 = quat2mat(camQuat3)

depth2cam1 = np.dot(rotMat1,depths1.transpose()) + camXYZ1.reshape((3,1))
depth2cam2 = np.dot(rotMat2,depths2.transpose()) + camXYZ2.reshape((3,1))
depth2cam3 = np.dot(rotMat3,depths3.transpose()) + camXYZ3.reshape((3,1))

#print depths1.shape
#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#ax.scatter(depths1[:,0],depths1[:,1],depths1[:,2],s=2,c='r')
#ax.scatter(depths2[:,0],depths2[:,1],depths2[:,2],s=2,c='b')
#ax.scatter(depth2cam1[0,:],depth2cam1[1,:],depth2cam1[2,:],s=5,c='b')
#ax.scatter(depth2cam2[0,:],depth2cam2[1,:],depth2cam2[2,:],s=5,c='r')

#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#plt.show()

#np.savetxt('export/depth2cam3.txt',depth2cam3.transpose(),delimiter='\t')
#np.savetxt('export/depth2cam1.txt',depth2cam1.transpose(),delimiter='\t')
#np.savetxt('export/depth2cam2.txt',depth2cam2.transpose(),delimiter='\t')

#np.savetxt('export/depths1.txt',depths1,delimiter='\t')
#np.savetxt('export/depths2.txt',depths2,delimiter='\t')
#np.savetxt('export/depths3.txt',depths3,delimiter='\t')