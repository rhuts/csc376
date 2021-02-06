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
# scipy (python package)
# numpy (python package)
# dijkstar (python package)

import b0RemoteApi
import math
import numpy as np
import random
import itertools
from Node import Node
from scipy.spatial import cKDTree
from dijkstar import Graph, find_path
import time
import argparse

# ----------- GLOBALS ------------\
# Sample size
N = 4

# roadmap
# size = (2 + number of collision free samples)
R = []

# kd tree data
kdData = np.zeros((2, 6))

# number of nearest neighbors to check
# will be clamped with total number of nodes
K_NEIGHBORS = 40

# local distance interpolations
N_INTERPOL = 20

# set to print debug msgs
DEBUG = False

# array to store all the configs of each colored ball
position_configs = []

# red ball
goal_angle1 = -111.8*math.pi/180
goal_angle2 = 43.8*math.pi/180
goal_angle3 = -92*math.pi/180
goal_angle4 = 173.5*math.pi/180
goal_angle5 = -390*math.pi/180
goal_angle6 = -142.4*math.pi/180

red = [goal_angle1, goal_angle2, goal_angle3, goal_angle4, goal_angle5, goal_angle6]
position_configs.append(red)

# green ball
goal_angle1 = -59.4*math.pi/180
goal_angle2 = 53.4*math.pi/180
goal_angle3 = -55*math.pi/180
goal_angle4 = 204.6*math.pi/180
goal_angle5 = -445*math.pi/180
goal_angle6 = -348.1*math.pi/180

green = [goal_angle1, goal_angle2, goal_angle3, goal_angle4, goal_angle5, goal_angle6]
position_configs.append(green)

# blue ball
goal_angle1 = -90.9*math.pi/180
goal_angle2 = -1.7*math.pi/180
goal_angle3 = 76.6*math.pi/180
goal_angle4 = 126.2*math.pi/180
goal_angle5 = -110.8*math.pi/180
goal_angle6 = -42.19*math.pi/180

blue = [goal_angle1, goal_angle2, goal_angle3, goal_angle4, goal_angle5, goal_angle6]
position_configs.append(blue)

# yellow ball
goal_angle1 = -83.2*math.pi/180
goal_angle2 = -59.6*math.pi/180
goal_angle3 = 87.9*math.pi/180
goal_angle4 = 413.1*math.pi/180
goal_angle5 = -252.8*math.pi/180
goal_angle6 = -246.6*math.pi/180

yellow = [goal_angle1, goal_angle2, goal_angle3, goal_angle4, goal_angle5, goal_angle6]
position_configs.append(yellow)

# purple ball
goal_angle1 = 6.55*math.pi/180
goal_angle2 = -14.7*math.pi/180
goal_angle3 = -74*math.pi/180
goal_angle4 = 618.1*math.pi/180
goal_angle5 = -60.1*math.pi/180
goal_angle6 = -573.3*math.pi/180

purple = [goal_angle1, goal_angle2, goal_angle3, goal_angle4, goal_angle5, goal_angle6]
position_configs.append(purple)

# ------- END GLOBALS ------------/




# Move joints to angles defined by config vector
def setJoints(config):

    global joint1Handle
    global joint2Handle
    global joint3Handle
    global joint4Handle
    global joint5Handle
    global joint6Handle

    # unfold config into angles
    angle1, angle2, angle3, angle4, angle5, angle6 = config

    #Set angles as targets for joint controllers (joint offsets are considered to match the V-REP home position)
    client.simxSetJointPosition(joint1Handle,angle1 - math.pi/2,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint2Handle,angle2 + math.pi,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint3Handle,angle3 + math.pi,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint4Handle,angle4,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint5Handle,angle5,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint6Handle,angle6,client.simxDefaultPublisher())

# Reset joints back to zero/home position
def resetJoints():

    global joint1Handle
    global joint2Handle
    global joint3Handle
    global joint4Handle
    global joint5Handle
    global joint6Handle

    #Reset joint angles to initial values
    client.simxSetJointPosition(joint1Handle,-math.pi/2,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint2Handle,math.pi,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint3Handle,math.pi,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint4Handle,0,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint5Handle,0,client.simxDefaultPublisher())
    client.simxSetJointPosition(joint6Handle,0,client.simxDefaultPublisher())

# Checks whether the input robot configuration is collision free and returns true if no
# collision is detected and false otherwise
def isCollisionFree(config):

    global obstaclesHandle
    global micoHandle
    
    setJoints(config)

    succ, res = client.simxCheckCollision(obstaclesHandle, micoHandle, client.simxServiceCall())
    if not succ: print('failed to check collision')

    # print('{}'.format('[ YES collision ]' if res else '[ NO collision ]'))
    # resetJoints()
    return not res


# Returns a randomly sampled robot configuration respecting its joint limits
def sampleRobotConfiguration():
    # joint limits are:
    # 2 -> [-2.32, 2.31]
    # 3 -> [-2.81, 2.80]
    joint1 = random.uniform(-math.pi, math.pi)
    joint2 = random.uniform(-2.31, 2.30)
    joint3 = random.uniform(-2.81, 2.80)
    joint4 = random.uniform(-math.pi, math.pi)
    joint5 = random.uniform(-math.pi, math.pi)
    joint6 = random.uniform(-math.pi, math.pi)

    return np.array([joint1, joint2, joint3, joint4, joint5, joint6])
    

# Sends a sequence of robot configurations (determined by planMotion) to
# V-REP using the API    
def runConfigurationSequence(seq_config):

    global N_INTERPOL

    # start with first config
    last_cfg = seq_config[0]
    setJoints(last_cfg)
    time.sleep(0.2)       # delay for visualization

    # loop through the remaining configs
    for curr_cfg in seq_config[1:]:

        interpolations = getInterpolations(last_cfg, curr_cfg, num=N_INTERPOL)
        # for interpolations between last and current config
        for cfg in interpolations:

            #setJoints
            setJoints(cfg)
            global DEBUG
            if DEBUG:
                print('\tSetting cfg: {}'.format(cfg))
            time.sleep(0.2)       # delay for visualization
        
        # update last config
        last_cfg = curr_cfg

# Returns a (num x 6) 2D array of
# linearly interpolated configurations
def getInterpolations(a_cfg, b_cfg, num):
    return np.linspace(a_cfg, b_cfg, num)


# Finds the k nodes nearest to node q in the roadmap R
# Returns the distances and indices of the k neighbor nodes
def findNearest(i, k):

    # create kd-tree
    global kdData
    tree = cKDTree(kdData)

    # query kd-tree for nearest neighbors and their distances
    dd, ii = tree.query(kdData[i], k=k)
    return dd, ii

# Returns whether there is a collision-free local path from a to b
# in the roadmap R
def existsLocalPath(a_cfg, b_cfg):
    
    global N_INTERPOL

    # interpolate path from A to B
    interpolations = getInterpolations(a_cfg, b_cfg, num=N_INTERPOL)

    # check for collisions along the interpolated points
    collision_free = True
    for config in interpolations:

        if not isCollisionFree(config):
            collision_free = False
            break

    # print('{}'.format('[ YES existsLocalPath ]' if res else '[ NO existsLocalPath ]'))
    return collision_free

# Returns the A* shortest path
# in the graph roadmap R from start to goal
# start_ind     the index of the starting node in R
# goal_ind      the index of the goal node in R
# Requires 'pip install Dijkstar'
def shortestPathAStar(start_ind, goal_ind):

    global R
    # R[0] is the node for the start configuration
    # R[1] is the node for the goal configuration
    
    # each node has edges in the form: self.edges = [] List of tuples (node_id, dist)
    # loop through our graph and convert it to a nice format for dijkstar
    graph = Graph()
    for node in R:
        for edge in node.edges:
            graph.add_edge(node.id,edge[0],edge[1])

    cfg_array=[]

    # catch path not found exception
    try:
        pathinfo = find_path(graph, start_ind, goal_ind)

    except:
        print('Could NOT find a path from start to goal')
        return cfg_array


    # get the configurations from each node in this found path
    for node_id in pathinfo.nodes:
        cfg_array.append(R[node_id].cfg)

    return cfg_array

# Gets a start and goal robot configuration as input and determines a sequence of collision free
# robot configurations using the probabilistic complete motion planning method of your choice
# Returns the found sequence of collision free robot configurations
def planMotion(start_config, goal_config):

    # the first two phases generate the roadmap
    # while the last phase finds the shortest path
    # from the generated roadmap

    global R
    global kdData
    global K_NEIGHBORS

    # Add the root (start) node and end (goal) node
    R.append(Node(start_config, 0))
    kdData[0, :] = start_config
    R.append(Node(goal_config, 1))
    kdData[1, :] = goal_config

    
    # ============= first phase ===========
    # take N samples of the C space 
    # and add the collision free ones as nodes of the graph
    for i in range(2 + N):      # offset by start and goal nodes

        q_i = Node(sampleRobotConfiguration(), len(R))

        if isCollisionFree(q_i.cfg):

            R.append(q_i)
            kdData = np.vstack((kdData, q_i.cfg))

    global DEBUG
    if DEBUG:
        print('Built up data for kd-tree:\n')    
        print(kdData)
        print('\n\n')



    # ============= second phase =============
    # connect the nodes with nearest neighbors
    for R_ind in range(len(R)):

        print('Processing neighbors of Node {}/{} ...'.format(R_ind, len(R)))

        # find a set of k nearby nodes
        num_neighbors = min(len(R), K_NEIGHBORS)            # can't have more neighbors than there are nodes
        neighbors_dd, neighbors_jj = findNearest(R[R_ind].id, k=num_neighbors)

        # try to find a path from the original node to each neighbor
        for j_ind in range(len(neighbors_jj)):

            neighbor_dist, neighbor_id = neighbors_dd[j_ind], neighbors_jj[j_ind]

            # no looping edges
            if neighbor_id != R[R_ind].id:

                # if there is a collision-free local path from original cfg to neighbor cfg
                if existsLocalPath(R[R_ind].cfg, R[neighbor_id].cfg):

                    # add an edge from original to neighbor if it doesn't already exist    
                    R[R_ind].addEdge(neighbor_id, neighbor_dist)

    
    # we now have the roadmap complete
    global DEBUG
    if DEBUG:
        print('Built up nodes w/o edges in R:\n')
        for node in R:
            print('{}\n'.format(node))


    # ============= third phase ===========
    # finds the shortest path from the roadmap
    # use Dijkstra's to find shortest path from start to goal
    path = shortestPathAStar(0, 1)

    return path

# Prints information for the
# current Probabilistic Roadmap configuration
def printPRMConfig():
    global N
    global K_NEIGHBORS
    global N_INTERPOL
    global DEBUG
    print('/==================================\\')
    print('\t\tPRM Config')
    print('\tSample size N \t: {}'.format(N))
    print('\tK_NEIGHBORS \t: {}'.format(K_NEIGHBORS))
    print('\tN_INTERPOL \t: {}'.format(N_INTERPOL))
    print('\tDEBUG \t\t: {}'.format(DEBUG))
    print('\\==================================/\n\n\n')


with b0RemoteApi.RemoteApiClient('b0RemoteApi_pythonClient','b0RemoteApi') as client:    
    doNextStep=True
    
    #Create callback functions for certain events
    def simulationStepStarted(msg):
        simTime=msg[1][b'simulationTime']
        print('Simulation step started. Simulation time: ',simTime)
        
    def simulationStepDone(msg):
        simTime=msg[1][b'simulationTime']
        print('Simulation step done. Simulation time: ',simTime)
        global doNextStep
        doNextStep=True
        
    client.simxSynchronous(True)
    
    #Register callbacks
    client.simxGetSimulationStepStarted(client.simxDefaultSubscriber(simulationStepStarted))
    client.simxGetSimulationStepDone(client.simxDefaultSubscriber(simulationStepDone))

    # Print PRM configs
    printPRMConfig()

    
    # handle args
    parser = argparse.ArgumentParser()
    parser.add_argument('start', type=int)
    parser.add_argument('goal', type=int)
    args = parser.parse_args()

    assert args.start > 0, 'Available positions are: [1, 5]'
    assert args.start < 6, 'Available positions are: [1, 5]'
    assert args.goal > 0, 'Available positions are: [1, 5]'
    assert args.goal < 6, 'Available positions are: [1, 5]'

    print('/----------------------------------\\')
    print('\tStart position\t: {}'.format(args.start))
    print('\tGoal position\t: {}'.format(args.goal))
    print('\\----------------------------------/\n')
    
    
    #Get joint handles
    error,joint1Handle=client.simxGetObjectHandle('Mico_joint1',client.simxServiceCall())
    error,joint2Handle=client.simxGetObjectHandle('Mico_joint2',client.simxServiceCall())
    error,joint3Handle=client.simxGetObjectHandle('Mico_joint3',client.simxServiceCall())
    error,joint4Handle=client.simxGetObjectHandle('Mico_joint4',client.simxServiceCall())
    error,joint5Handle=client.simxGetObjectHandle('Mico_joint5',client.simxServiceCall())
    error,joint6Handle=client.simxGetObjectHandle('Mico_joint6',client.simxServiceCall())
    
    # Get robot and obstacle collection handles
    succ, obstaclesHandle = client.simxGetCollectionHandle('myObstacles#0', client.simxServiceCall())
    if not succ: print('failed to get obstacles!')

    succ, micoHandle = client.simxGetCollectionHandle('myMicoLinks#0', client.simxServiceCall())
    if not succ: print('failed to get mico!')

    #Define desired joint angles in rad

    # free config
    angle1 = -111.8*math.pi/180
    angle2 = 43.8*math.pi/180
    angle3 = -60*math.pi/180
    angle4 = 173.5*math.pi/180
    angle5 = -390*math.pi/180
    angle6 = -142.4*math.pi/180
    free_cfg = [angle1,angle2,angle3,angle4,angle5,angle6]

    # collision config
    angle1 = -111.8*math.pi/180
    angle2 = 43.8*math.pi/180
    angle3 = -112*math.pi/180
    angle4 = 173.5*math.pi/180
    angle5 = -390*math.pi/180
    angle6 = -142.4*math.pi/180
    collision_cfg = [angle1,angle2,angle3,angle4,angle5,angle6]



    # simple test for taking random samples and checking for collisions
    TEST_SAMPLE_AND_COLLISION = 0
    if TEST_SAMPLE_AND_COLLISION:

        # Check for collision
        config = [angle1,angle2,angle3,angle4,angle5,angle6]
        isCollisionFree(config)

        isCollisionFree([0, 0, 0, 0, 0, 0])

        # Check 10 random samples for collision
        for i in range(0, 10):
            isCollisionFree(sampleRobotConfiguration())



    # simple test for runConfigurationSequence
    TEST_RUN_CONFIG_SEQUENCE = 0
    if TEST_RUN_CONFIG_SEQUENCE:
        # move robot through random sequence
        raw_input('before runConfigurationSequence()')
        seq = [sampleRobotConfiguration() for i in range(10)]
        runConfigurationSequence(seq)


    # simple test for local path planner
    TEST_EXISTS_LOCAL_PATH = 0
    if TEST_EXISTS_LOCAL_PATH:
        
        setJoints(free_cfg)
        raw_input('moving to collision manually ...')
        setJoints(collision_cfg)
        raw_input('moved ...')
        resetJoints()
        raw_input('reset, about to check ...')
        existsLocalPath(free_cfg, collision_cfg)

        

    # setting TEST_PLAN_MOTION
    # performs path planning to the uncommented goal_angles
    # and carries out the computed path
    TEST_PLAN_MOTION = 1
    if TEST_PLAN_MOTION:
        # Plan motion
        
        start = position_configs[args.start - 1]
        goal = position_configs[args.goal - 1]
        path = []

        while not path:
            global R
            global kdData
            R = []
            kdData = np.zeros((2, 6))
            path = planMotion(start, goal)
            if path:
                print('\nFOUND! path from start to goal configuration')
                resetJoints(); raw_input('Press a key to move from start to goal configuration! ...')
                runConfigurationSequence(path)
                raw_input('Press a key to reset to home configuration! ...')
            else:
                print('\nFailed to find path from start to goal configuration')

    
    #Reset joint angles to initial values
    resetJoints()