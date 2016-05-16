__author__ = 'sonal'
import numpy as np
import sys
sys.path.insert(0,'/usr/local/lib/python2.7/site-packages/')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import *
'''
depth1 = "data/May 12/depth-461.535240001.txt"
depth2 = "data/May 12/depth-470.185728001.txt"
depth3 = "data/may 12/depth-480.015828001.txt"
pose1 = "data/May 12/pose-461.535240001.txt"
pose2 = "data/May 12/pose-470.185728001.txt"
pose3 = "data/May 12/pose-480.015828001.txt"
'''

depth1 = "data/May 16/depth-146.889319001.txt"
depth2 = "data/May 16/depth-150.591990001.txt"
#depth3 = "data/May 16/depth-152.099272001.txt"
depth3 = "data/May 16/depth-155.670875001.txt"
pose1 = "data/May 16/pose-146.889319001.txt"
pose2 = "data/May 16/pose-150.591990001.txt"
#pose3 = "data/May 16/pose-152.099272001.txt"
pose3 = "data/May 16/pose-155.670875001.txt"
#camXYZ1 = np.array([0.547282388237575, 0.157351187292134, -0.0254139935371636]).transpose()
#camQuat1 = np.array([0.501725122125409, -0.240603849434341, -0.353766788261331, 0.751818295194313])
#camXYZ2 = np.array([0.541529891261503,0.156750238065981,-0.0181067783655014]).transpose()
#camQuat2 = np.array([0.513630923400241,-0.243933364155223,-0.348845268070899,0.744974340043651])
#camXYZ3 = np.array([0.554824368194775,0.165557872552343,0.00639985921924491]).transpose()
#camQuat3 = np.array([0.482640892361687,-0.26910679415059,-0.378285271369796,0.742657091682955])


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
            x = float(element[3][1:len(element[3])])
            y = float(element[4])
            z = float(element[5][0:(len(element[5]) - 1)])
            q1 = float(element[6][2:len(element[6])])
            q2 = float(element[7])
            q3 = float(element[8])
            q4 = float(element[9][0:(len(element[9]) - 1)])

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



    rotMat = np.array([[q11, q12, q13],
                       [q21, q22, q23],
                       [q31, q32, q33]])

    return rotMat

def mat2quat(mat):
    tr = mat[0, 0] + mat[1, 1] + mat[2, 2]

    if (tr > 0):
        S = sqrt(tr + 1.0) * 2 # S= 4*q1
        qw = 0.25 * S
        qx = (mat[2, 1] - mat[1, 2]) / S
        qy = (mat[0, 2] - mat[2, 0]) / S
        qz = (mat[1, 0] - mat[0, 1]) / S
    elif ((mat[0, 0] > mat[1, 1]) and (mat[0, 0] > mat[2, 2])):
        S = sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2 # S = 4*qx
        qw = (mat[2,1 ] - mat[1, 2]) / S
        qx = 0.25 * S
        qy = (mat[0, 1] + mat[1, 0]) / S
        qz = (mat[0, 2] + mat[2, 0]) / S
    elif (mat[1, 1] > mat[2, 2]):
        S = sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2 # S = 4*qy
        qw = (mat[0, 2] - mat[2, 0]) / S
        qx = (mat[0, 1] + mat[1, 0]) / S
        qy = 0.25 * S
        qz = (mat[1, 2] + mat[2, 1]) / S

    else:
        S = sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2 # S=4*qz
        qw = (mat[1, 0] - mat[0, 1]) / S
        qx = (mat[0, 2] - mat[2, 0]) / S
        qy = (mat[1, 2] - mat[2, 1]) / S
        qz = 0.25 * S
    quat = np.hstack((((qw,qx),qy),qz))
    return quat





camXYZ1 = parsePoseQuat(pose1)[0]
camXYZ2 = parsePoseQuat(pose2)[0]
camXYZ3 = parsePoseQuat(pose3)[0]

depths1 = parseDepth(depth1)
rotMat1 = quat2mat(parsePoseQuat(pose1)[1])
depths2 = parseDepth(depth2)
rotMat2 = quat2mat(parsePoseQuat(pose2)[1])
depths3 = parseDepth(depth3)
rotMat3 = quat2mat(parsePoseQuat(pose3)[1])

depth2cam1 = np.dot(rotMat1, depths1.transpose()) + camXYZ1.reshape((3, 1))
depth2cam2 = np.dot(rotMat2, depths2.transpose()) + camXYZ2.reshape((3, 1))
depth2cam3 = np.dot(rotMat3, depths3.transpose()) + camXYZ3.reshape((3, 1))

depths12 = np.vstack((depth2cam1.transpose(),depth2cam2.transpose()))
depths123 = np.vstack((depths12,depth2cam3.transpose()))
#np.savetxt('export/May_12_depths.txt',depths123,delimiter='\t')


camxyz1 = camXYZ1.reshape((3,1))
camxyz2 = camXYZ2.reshape((3,1))
camxyz3 = camXYZ3.reshape((3,1))

pose_x = np.vstack(((camxyz1[0],camxyz2[0]),camxyz3[0]))
pose_y = np.vstack(((camxyz1[1],camxyz2[1]),camxyz3[1]))
pose_z = np.vstack(((camxyz1[2],camxyz2[2]),camxyz3[2]))

labels = ['1','2','3']

matRot3 = mat2quat(rotMat3)
print parsePoseQuat(pose3)[1]
print matRot3
'''
plt.scatter(depths123[::5,0],depths123[::5,2])

#for i, txt in enumerate(labels):
#    plt.annotate(txt,(pose_x[i],pose_y[i]))
plt.show()'''

#print depths1.shape
'''
fig = plt.figure(1)
ax = fig.add_subplot(111,projection='3d')

#ax.scatter(depths123[::10,0],depths123[::10,1], depths123[::10,2],s=2)
#ax.scatter(depths123[::10, 0], depths123[::10, 1], depths123[::10, 2], s=1, c='b')
ax.scatter(depths1[::5,0], depths1[::5,1], depths1[::5,2],s=1)
ax.pbaspect=[1.0,1.0,1.0]
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
'''
'''
fig = plt.figure(2)
ax = fig.add_subplot(111,projection='3d')

#ax.scatter(depths123[::10,0],depths123[::10,1], depths123[::10,2],s=2)
ax.scatter(depths2[::10,0],depths2[::10,1], depths2[::10,2],s=2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()'''

#np.savetxt('export/depth2cam3.txt',depth2cam3.transpose(),delimiter='\t')
#np.savetxt('export/depth2cam1.txt',depth2cam1.transpose(),delimiter='\t')
#np.savetxt('export/depth2cam2.txt',depth2cam2.transpose(),delimiter='\t')

#np.savetxt('export/depths1.txt',depths1,delimiter='\t')
#np.savetxt('export/depths2.txt',depths2,delimiter='\t')
#np.savetxt('export/depths3.txt',depths3,delimiter='\t')
#np.savetxt('export/garbage-can1.txt',depths123, delimiter='\t')