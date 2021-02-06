# Copyright Continuum Robotics Laboratory, UTM, Canada, 2019
#
# Before running this script make sure you have loaded the scene "AssignmentII.ttt" in V-REP!
#
# The script depends on:
#
# b0RemoteApi (Python script), which depends on:
# msgpack (Python messagePack binding, install with "pip install msgpack")
# b0.py (Python script), which depends on:
# b0 (shared library), which depends on:
# boost_chrono (shared library)
# boost_system (shared library)
# boost_thread (shared library)
# libzmq (shared library)

import b0RemoteApi
import math
import numpy as np


# ----------- GLOBALS ------------\

D1 = 0.2755
D2 = 0.2900
D3 = 0.1233
D4 = 0.0741
D5 = 0.0741
D6 = 0.1600
e2 = 0.0070

aa = 30.0*math.pi/180
sa = np.sin(aa)
s2a = np.sin(2 * aa)
d4b = D3 + ((sa/s2a) * D4)
d5b = ((sa/s2a) * D4) + ((sa/s2a) * D5)
d6b = ((sa/s2a) * D4) + D6
c2a = np.sin(2 * aa)
ca = np.cos(aa)

# ------- END GLOBALS ------------/




# Takes     e^[S_1]theta_1
#
# which is exponential coordinate
#
# Returns     R     p
#           0 0 0   1
#
# which is 4 x 4 transformation matrix
def rodrigues(theta, w, v):

    # Calculate R matrix
    I = np.identity(3)
    second = np.sin(theta) * w
    third = (1 - np.cos(theta)) * (w**2)

    R = I + second + third



    # Calculate p vector
    one = np.identity(3) * theta
    two = (1 - np.cos(theta)) * w
    three = (theta - np.sin(theta)) * (w**2)

    p = np.matmul((one + two + three), v)



    # Concatenate R matrix with p vector and bottom row
    Rp = np.hstack((R, p))
    T = np.vstack((Rp, [0, 0, 0, 1]))

    return T

# This function takes a list containing actuation angles and returns a matrix containing the end-effector pose
def forwardKinematicsPEF(angles):
    print('angles:')
    print(angles)
    print('')

    w1 = np.matrix([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 0]])

    v1 = np.matrix([[0],
                    [0],
                    [0]])

    w2 = np.matrix([[0, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]])
                    
    v2 = np.matrix([[0],
                    [D1],
                    [0]])

    w3 = np.matrix([[0, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]])
                    
    v3 = np.matrix([[0],
                    [-(D1 + D2)],
                    [0]])

    w4 = np.matrix([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 0]])
                    
    v4 = np.matrix([[0],
                    [e2],
                    [0]])

    w5 = np.matrix([[0, sa, 0],
                    [-sa, 0, -ca],
                    [0, ca, 0]])
                    
    v5 = np.matrix([[0],
                    [(sa * e2) + ca * (D1 + D2 +d4b)],
                    [0]])

    w6 = np.matrix([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 0]])
                    
    v6 = np.matrix([[0],
                    [e2 - (d5b * s2a)],
                    [0]])

    M = np.matrix([ [1, 0, 0, e2 - (d5b * s2a)],
                    [0, 1, 0, 0],
                    [0, 0, 1, D1 + D2 + d4b + (d5b * c2a) + d6b],
                    [0, 0, 0, 1]])

    T1 = rodrigues(angles[0], w1, v1)

    T2 = rodrigues(angles[1], w2, v2)

    T3 = rodrigues(angles[2], w3, v3)

    T4 = rodrigues(angles[3], w4, v4)

    T5 = rodrigues(angles[4], w5, v5)

    T6 = rodrigues(angles[5], w6, v6)

    Tfinal = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(T1, T2), T3), T4), T5), T6), M)

    return Tfinal

def forwardKinematicsDH(angles):

    R1 = np.matrix([[0, 1, 0, 0],
                    [-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    R2 = np.matrix([[-1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])

    T01 = np.matrix([[1, 0, 0, 0],[0, 0, -1, -D1],[0, 1, 0, 0],[0, 0, 0, 1]])

    T12 = np.matrix([[1, 0, 0, D2],[0, -1, 0, 0],[0, 0, -1, 0],[0, 0, 0, 1]])

    T23 = np.matrix([[1, 0, 0, 0],[0, 0, -1, e2],[0, 1, 0, 0],[0, 0, 0, 1]])

    T34 = np.matrix([[1, 0, 0, 0],[0, 0.5, -math.sqrt(3)/2, d4b*(math.sqrt(3)/2)],[0, math.sqrt(3)/2, 0.5, -d4b*0.5],[0, 0, 0, 1]])

    T45 = np.matrix([[1, 0, 0, 0],[0, 0.5, -math.sqrt(3)/2, d5b*(math.sqrt(3)/2)],[0, math.sqrt(3)/2, 0.5, -d5b*0.5],[0, 0, 0, 1]])

    T56 = np.matrix([[0, 1, 0, 0],[1, 0, 0, 0],[0, 0, -1, d6b],[0, 0, 0, 1]])

    T = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(R1, R2), T01), T12), T23), T34), T45), T56)

    return T


# Takes         p   which is 3 x 1
#
# Returns       [p] which is 3 x 3
def skew(p):

    a1 = p[0, 0]
    a2 = p[1, 0]
    a3 = p[2, 0]

    skew = np.matrix([[0, -a3, a2],
                     [a3, 0, -a1],
                     [-a2, a1, 0]])

    return skew

# Takes         T in R 4 x 4
# [ R p ]
# [ 0 1 ]
#
# Returns       Ad_T in R 6 x 6
# [ R       0 ]
# [ [p]R    R ]
def adjoint(T):

    # extract R and p from T
    R = np.delete(np.delete(T, 3, 0), 3, 1)
    p = T[[0,1,2],3].reshape((3, 1))

    zeros = np.zeros(9).reshape((3, 3))
    pR = np.matmul(skew(p), R)

    # concatenate matricies into Ad_T
    top = np.hstack((R, zeros))
    bot = np.hstack((pR, R))

    Ad_T = np.vstack((top, bot))

    return Ad_T
    
# This function takes a list containing actuation angles and returns the jacobian matrix
def velocityKinematics(angles):

    w1 = np.matrix([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 0]])

    v1 = np.matrix([[0],
                    [0],
                    [0]])

    w2 = np.matrix([[0, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]])
                    
    v2 = np.matrix([[0],
                    [D1],
                    [0]])

    w3 = np.matrix([[0, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]])
                    
    v3 = np.matrix([[0],
                    [-(D1 + D2)],
                    [0]])

    w4 = np.matrix([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 0]])
                    
    v4 = np.matrix([[0],
                    [e2],
                    [0]])

    w5 = np.matrix([[0, sa, 0],
                    [-sa, 0, -ca],
                    [0, ca, 0]])
                    
    v5 = np.matrix([[0],
                    [(sa * e2) + ca * (D1 + D2 +d4b)],
                    [0]])

    w6 = np.matrix([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 0]])
                    
    v6 = np.matrix([[0],
                    [e2 - (d5b * s2a)],
                    [0]])

    M = np.matrix([ [1, 0, 0, e2 - (d5b * s2a)],
                    [0, 1, 0, 0],
                    [0, 0, 1, D1 + D2 + d4b + (d5b * c2a) + d6b],
                    [0, 0, 0, 1]])

    T1 = rodrigues(angles[0], w1, v1)

    T2 = rodrigues(angles[1], w2, v2)

    T3 = rodrigues(angles[2], w3, v3)

    T4 = rodrigues(angles[3], w4, v4)

    T5 = rodrigues(angles[4], w5, v5)

    T6 = rodrigues(angles[5], w6, v6)

    S1 = np.array([0, 0, -1, 0, 0, 0]).reshape((6, 1))
    S2 = np.array([1, 0, 0, 0, D1, 0]).reshape((6, 1))
    S3 = np.array([-1, 0, 0, 0, -(D1 + D2), 0]).reshape((6, 1))
    S4 = np.array([0, 0, -1, 0, e2, 0]).reshape((6, 1))
    S5 = np.array([ca, 0, -sa, 0, (sa * e2) + ca * (D1 + D2 + d4b), 0]).reshape((6, 1))
    S6 = np.array([0, 0, -1, 0, e2 - (d5b * s2a), 0]).reshape((6, 1))

    jacobian = np.matrix([[0],
                          [0],
                          [1],
                          [0],
                          [0],
                          [0]])
    
    # Js2 #######################

    # get adjoint mapping of Ti-1
    AdT1 = adjoint(T1)

    # multiply by Si
    Js2 = np.matmul(AdT1, S2)

    # fill in i-th column of jacobian
    jacobian = np.hstack((jacobian, Js2))


    # Js3 #######################

    # get adjoint mapping of Ti-1
    AdT2 = adjoint(np.matmul(T1, T2))

    # multiply by Si
    Js3 = np.matmul(AdT2, S3)

    # fill in i-th column of jacobian
    jacobian = np.hstack((jacobian, Js3))


    # Js4 #######################

    # get adjoint mapping of Ti-1
    AdT3 = adjoint(np.matmul(np.matmul(T1, T2), T3))

    # multiply by Si
    Js4 = np.matmul(AdT3, S4)

    # fill in i-th column of jacobian
    jacobian = np.hstack((jacobian, Js4))


    # Js5 #######################

    # get adjoint mapping of Ti-1
    AdT4 = adjoint(np.matmul(np.matmul(np.matmul(T1, T2), T3), T4))

    # multiply by Si
    Js5 = np.matmul(AdT4, S5)

    # fill in i-th column of jacobian
    jacobian = np.hstack((jacobian, Js5))


    # Js6 #######################

    # get adjoint mapping of Ti-1
    AdT5 = adjoint(np.matmul(np.matmul(np.matmul(np.matmul(T1, T2), T3), T4), T5))

    # multiply by Si
    Js6 = np.matmul(AdT5, S6)

    # fill in i-th column of jacobian
    jacobian = np.hstack((jacobian, Js6))
    
    
    return jacobian

# This function takes the jacobian matrix and returns the reciprocal condition number
def calcRCond(jacobian):

    # Compute JJ^T
    A = np.matmul(jacobian, jacobian.T)

    # Get eigenvalues
    w, v = np.linalg.eig(A)
    
    # Calculate rCond
    rcond = 1 / (np.max(w) / np.min(w))
    
    return rcond

# This function takes the jacobian matrix and returns the singular vectors and values associated with the end-effector position\
# Note: You are allowed to use the SVD method from numpy!
def calcSingularValues(jacobian):
    
    
    #Placeholder, replace by your calculation
    sigma_1 = 0.1
    sigma_2 = 0.1
    sigma_3 = 0.1
    
    u_1 = np.matrix([1,0,0])
    u_2 = np.matrix([0,1,0])
    u_3 = np.matrix([0,0,1])
    
    
    return u_1, u_2, u_3, sigma_1, sigma_2, sigma_3

#This function takes the three singular vectors and values associated with the end-effector position and draws the manipulability
#ellipsoid. The handles of the drawn objects should be returned to delete them later in the code, when the scene is reset.
def drawManEllipsoid(u_1,u_2,u_3,sigma_1,sigma_2,sigma_3,ee_pose):
    
    # Added
    print('drawing manipulability ellipsoid ... ')
    print('')

    list_handles = []
            
    return list_handles

with b0RemoteApi.RemoteApiClient('b0RemoteApi_pythonClient','b0RemoteApi') as client:    
    doNextStep=True
    
    #Create callback functions for certain events
    def simulationStepStarted(msg):
        simTime=msg[1][b'simulationTime'];
        print('Simulation step started. Simulation time: ',simTime)
        
    def simulationStepDone(msg):
        simTime=msg[1][b'simulationTime'];
        print('Simulation step done. Simulation time: ',simTime);
        global doNextStep
        doNextStep=True
        
    client.simxSynchronous(True)
    
    #Register callbacks
    client.simxGetSimulationStepStarted(client.simxDefaultSubscriber(simulationStepStarted))
    client.simxGetSimulationStepDone(client.simxDefaultSubscriber(simulationStepDone))
    
    #Get joint handles
    error,joint1Handle=client.simxGetObjectHandle('Mico_joint1',client.simxServiceCall())
    error,joint2Handle=client.simxGetObjectHandle('Mico_joint2',client.simxServiceCall())
    error,joint3Handle=client.simxGetObjectHandle('Mico_joint3',client.simxServiceCall())
    error,joint4Handle=client.simxGetObjectHandle('Mico_joint4',client.simxServiceCall())
    error,joint5Handle=client.simxGetObjectHandle('Mico_joint5',client.simxServiceCall())
    error,joint6Handle=client.simxGetObjectHandle('Mico_joint6',client.simxServiceCall())
    
    #Get end-effector handle
    error,eeHandle=client.simxGetObjectHandle('MicoHand_Dummy2',client.simxServiceCall())
    
    #Define desired joint angles in rad
    angle1 = 90*math.pi/180
    angle2 = -60*math.pi/180
    angle3 = 40*math.pi/180
    angle4 = 110*math.pi/180
    angle5 = 150*math.pi/180
    angle6 = 50*math.pi/180
    
    #Calculate forward kinematics based on implemented functions (these functions needs to be implemented!!)
    eePoseFK_PEF = forwardKinematicsPEF([angle1,angle2,angle3,angle4,angle5,angle6])
    eePoseFK_DH = forwardKinematicsDH([angle1,angle2,angle3,angle4,angle5,angle6])


    # Added
    print('eePose forwardKinematicsPEF:')
    print(eePoseFK_PEF)
    print('')
    print('eePose forwardKinematicsDH:')
    print(eePoseFK_DH)
    print('')
    

    
    #Get end-effector position from pose
    # eePositionFK = [eePoseFK[0,3],eePoseFK[1,3],eePoseFK[2,3]]
    eePositionFK = [eePoseFK_PEF[0,3],eePoseFK_PEF[1,3],eePoseFK_PEF[2,3]]  # Added
    # eePositionFK = [eePoseFK_DH[0,3],eePoseFK_DH[1,3],eePoseFK_DH[2,3]]   # Added
    
    #Draw calculated end-effector position as a blue sphere in V-REP
    error, visEEHandle = client.simxAddDrawingObject_spheres(0.025,[0,0,125],[eePositionFK[0],eePositionFK[1],eePositionFK[2]],client.simxServiceCall())
    
    #Set angles as targets for joint controllers (joint offsets are considered to match the V-REP home position)
    client.simxSetJointPosition(joint1Handle,angle1 - math.pi/2,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint2Handle,angle2 + math.pi,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint3Handle,angle3 + math.pi,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint4Handle,angle4,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint5Handle,angle5,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint6Handle,angle6,client.simxDefaultPublisher())
    
    #Wait for user input
    raw_input('Press Enter to continue...')
    
    #Get end-effector pose from V-REP 
    error,eePose=client.simxGetObjectPose(eeHandle,-1,client.simxServiceCall())
    
    #Transform obtained pose in 4x4 frame instead of position with quaternion
    s = 1/(eePose[3]*eePose[3] + eePose[4]*eePose[4] + eePose[5]*eePose[5] + eePose[6]*eePose[6])
    qr = eePose[6]
    qi = eePose[3]
    qj = eePose[4]
    qk = eePose[5]
    
    eePose = np.matrix([[1 - 2*s*(qj*qj + qk*qk), 2*s*(qi*qj - qk*qr),     2*s*(qi*qk + qj*qr),     eePose[0]],
              [2*s*(qi*qj + qk*qr),     1 - 2*s*(qi*qi + qk*qk), 2*s*(qj*qk - qi*qr),     eePose[1]],
              [2*s*(qi*qk - qj*qr),     2*s*(qj*qk + qi*qr),     1 - 2*s*(qi*qi + qj*qj), eePose[2]],
              [0,                       0,                       0,                       1]])

    print('Simulated robot eePose:')
    print(eePose)
    print('')
    
    
    #Calculate velocity kinematics based on implemented function (this function needs to be implemented!!)
    jacobian = velocityKinematics([angle1,angle2,angle3,angle4,angle5,angle6])
    # print('jacobian:')
    # print(jacobian)
    
    #Calculate reciprocal condition number for the obtained Jacobian matrix here (this function needs to be implemented!!)
    rcond = calcRCond(jacobian)

    # Added
    print('reciprocal conditional number:')
    print(rcond)
    print('')
    
    #Visualize manipulability ellipsoid w.r.t. end-effector position (not pose!) in the V-REP scene
    #In order to do so call the predefined function drawManEllipsoid with the three singular vectors and values calculated for the end-effector position
    #(This function needs to be implemented!)
    u_1,u_2,u_3,sigma_1,sigma_2,sigma_3 = calcSingularValues(jacobian)
    print('singular vectors and values:')
    print(u_1)
    print(u_2)
    print(u_3)
    print(sigma_1)
    print(sigma_2)
    print(sigma_3)
    print('')
    
    drawHandles = drawManEllipsoid(u_1,u_2,u_3,sigma_1,sigma_2,sigma_3,eePose)
    
    
    #Wait for user input
    raw_input('Press Enter to continue...')
    
    #Remove ellipsoid
    for i in drawHandles:
        client.simxRemoveDrawingObject(i, client.simxDefaultPublisher())
    
    #Remove drawn calculated end-effector position
    client.simxRemoveDrawingObject(visEEHandle,client.simxDefaultPublisher())
    
    #Reset joint angles to initial values
    client.simxSetJointPosition(joint1Handle,-math.pi/2,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint2Handle,math.pi,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint3Handle,math.pi,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint4Handle,0,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint5Handle,0,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint6Handle,0,client.simxDefaultPublisher())
    
