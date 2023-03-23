import os
import time

import pybullet as p
import gym
import numpy as np
import pybullet_data
# import math


from system.controller.navigationPhase import pick_intermediate_goal_vector, find_new_goal_vector
from system.helper import compute_angle
from system.bio_model.BoundaryCellModel import parametersBC
# from ..bio_model.BoundaryCellModel import parametersBC
# from ..controller.navigationPhase import pick_intermediate_goal_vector, find_new_goal_vector
# from ..helper import compute_angle

# PybulletEnvironment(gym.Env):
class PybulletEnvironment(gym.Env):
    def __init__(self, visualize, env_model,dt,pod=None):
        self.visualize = visualize  # to open JAVA application
        self.env_model = env_model  # string specifying env_model
        self.pod = pod
        #self.rate = rate
        self.dt=dt
        vis_cue_pos = [11, 11, 1]
        if self.visualize:
            physicsClient = p.connect(p.GUI)
            # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            physicsClient = p.connect(p.DIRECT)

        if self.env_model == 'maze':
            p.loadURDF("environment_map/maze_2_2_lane/plane.urdf")
            p.resetDebugVisualizerCamera(cameraDistance=4.5, cameraYaw=0, cameraPitch=-70,
                                         cameraTargetPosition=[0.55, -7.35, 5.0])
            basePosition = [8.0, -10, 0.02]
            vis_cue_pos = [8.5, -7.7, 0]
        elif self.env_model == 'curved':
            p.loadURDF("environment_map/maze_curved_elements/plane.urdf")
            p.resetDebugVisualizerCamera(cameraDistance=4.5, cameraYaw=0, cameraPitch=-70,
                                         cameraTargetPosition=[0.55, -7.35, 5.0])
            basePosition = [7.7, -10, 0.02]
        elif self.env_model == 'plus':
            p.loadURDF("p3dx/plane/plane.urdf")
            p.resetDebugVisualizerCamera(cameraDistance=4.5, cameraYaw=0, cameraPitch=-70,
                                         cameraTargetPosition=[0.55, -0.35, 0.2])
            basePosition = [-1.5, -2, 0.02]
            vis_cue_pos = [0, -2.7, 0]
        elif self.env_model == 'box':
            p.loadURDF("p3dx/plane/plane_box.urdf")
            p.resetDebugVisualizerCamera(cameraDistance=4.5, cameraYaw=0, cameraPitch=-70,
                                         cameraTargetPosition=[0.55, -0.35, 0.2])
            basePosition = [0, -0.5, 0.02]
            vis_cue_pos = [1.05, 0, 0]
        elif self.env_model == "obstacle":
            p.loadURDF("environment_map/obstacle_map/plane.urdf")
            p.resetDebugVisualizerCamera(cameraDistance=4.5, cameraYaw=0, cameraPitch=-70,
                                         cameraTargetPosition=[8, -4, 0.2])
            basePosition = [0, 0.05, 0.02]
            # self.cuePos = [0, -2.7, 0]
        elif self.env_model == "linear_sunburst":
            # doors option:"plane","plane_doors_open35","plane_doors_open15","plane_doors_open1","plane_doors_allopen"
            doors_option = "plane"  # "plane" for default, "plane_doors", "plane_doors_individual"
            self.door_option=doors_option
            p.loadURDF("environment_map/linear_sunburst_map/" + doors_option + ".urdf")
            # p.loadURDF('environment_map/linear_sunburst_map/plane_doors.urdf')
            # p.loadURDF("../../environment_map/linear_sunburst_map/" + doors_option + ".urdf")
            basePosition = [5.5, 0.55, 0.02]
            p.resetDebugVisualizerCamera(cameraDistance=4.5, cameraYaw=0, cameraPitch=-70,
                                         cameraTargetPosition=[5.5, 1.0, 5])
            # vis_cue_pos = [11, 11, 1]
            vis_cue_pos = [5, 5, 1]
            self.vis_cue_pos=vis_cue_pos #use self.for plot in plot subgoal
            arena_size = 15
            # goal_location = np.array([1.5, 10])
            max_speed = 6
        else:
            urdfRootPath = pybullet_data.getDataPath()
            p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"))
            p.resetDebugVisualizerCamera(cameraDistance=4.5, cameraYaw=0, cameraPitch=-70,
                                         cameraTargetPosition=[8, -4, 0.2])
            basePosition = [7.7, -10, 0.02]
            # goal_location = np.array([1.5, 10])
            # self.cuePos = [0, -2.7, 0]
            vis_cue_pos = [1250, 500, 0]

        orientation = p.getQuaternionFromEuler([0, 0, np.pi / 2])  # faces North
        # orientation = p.getQuaternionFromEuler([0, 0, 0])  # faces East
        self.cuePos = vis_cue_pos # also used in plot
        self.cueId = p.loadURDF("p3dx/cue/visual_cue.urdf", basePosition=self.cuePos)
        self.carId = p.loadURDF("p3dx/urdf/pioneer3dx.urdf", basePosition=basePosition, baseOrientation=orientation)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        # self.cuePos = cuePos  # position for landmark
        self.xy_coordinates = []  # keeps track of agent's coordinates at each time step
        self.orientation_angle = []  # keeps track of agent's orientation at each time step
        self.delta_distance=[]
        self.xy_speeds = []  # keeps track of agent's speed (vector) at each time step
        self.speeds = []  # keeps track of agent's speed (value) at each time step
        # self.save_position_and_speed()  # save initial configuration

        # # Set goal location to preset location or current position if none was specified
        # self.goal_location = goal_location if goal_location is not None else self.xy_coordinates[0]
        goal_location = np.array([1.5, 10])
        self.save_position_and_speed()  # save initial configuration
        self.goal_location = goal_location if goal_location is not None else self.xy_coordinates[0]
        self.max_speed = 6
        self.arena_size = 15
        self.goal = np.array([0, 0])  # used for navigation (eg. sub goals)

        self.goal_vector_original = np.array([1, 1])  # egocentric goal vector after last recalculation
        self.goal_vector = np.array([0, 0])  # egocentric goal vector after last update

        self.goal_idx = 0  # pc_idx of goal

        self.turning = False  # agent state, used for controller

        self.num_ray_dir = parametersBC.num_ray_16  # number of direction to check for obstacles for
        # self.num_ray_dir = 51  # number of direction to check for obstacles for
        self.num_travel_dir = parametersBC.num_travel_dir  # valid traveling directions, 4 -> [E, N, W, S]
        self.directions = np.empty(self.num_ray_dir, dtype=bool)  # array keeping track which directions are blocked
        self.topology_based = False  # agent state, used for controller
        self.time_elapsed = 0
        self.counter = 0
        self.v_L = 0
        self.v_R = 0
        self.planeId = []
        #self.cueId = []

        self.agentState = {'Position': None, 'Orientation': None}
        self.action_space = []
        self.observation_space = []
        self.euler_angle = 0
        #self.carId = []
        self.euler_angle_before = 0
        # addition for BC-Model helper that returns coordinates of encountered boundary segments wth getRays()
        self.raysThatHit = []

        self.euler_angle = p.getEulerFromQuaternion(p.getLinkState(self.carId, 0)[1])


        # p.setTimeStep(1.0 / self.rate)  # default is 240 Hz
       # goal_loacation = None



        # if env_model == "rw_circle":
        #     vis_cue_pos = [1850, 650, 0]  # rwrun 1
        # elif env_model == "rw_loops":
        #     vis_cue_pos = [1350, 400, 0]  # rw run 2
        # elif env_model == "rw_cross":
        #     vis_cue_pos = [1250, 500, 0]  # rw run 3
        # else:
        #     vis_cue_pos = [11, 11, 1]







    def compute_movement(self, gc_network, pc_network, cognitive_map, exploration_phase=True,currentHD=None):
        """Compute and set motor gains of agents. Simulate the movement with py-bullet"""
        # return v_left,v_right
        self.euler_angle_before=self.euler_angle
        # gains = self.avoid_obstacles(gc_network, pc_network, cognitive_map, exploration_phase)
        gains = self.avoid_obstacles(gc_network, pc_network, cognitive_map, exploration_phase,currentHD=currentHD)

        self.change_speed(gains) # appy v
        posAndOr=p.getBasePositionAndOrientation(self.carId)
        self.agentState['Position'] = posAndOr[0][:2]
        self.agentState['Orientation'] = p.getEulerFromQuaternion(posAndOr[1])[2]
        if (self.agentState['Orientation'] < 0): self.agentState['Orientation'] += 2*np.pi
        p.stepSimulation()
        self.save_position_and_speed() # update position,orientation,speed
        if self.visualize:
            time.sleep(self.dt/5)

        delta_distance=np.linalg.norm(self.xy_coordinates[-1]-self.xy_coordinates[-2])
        self.delta_distance.append(delta_distance)
        # c = np.linalg.norm(d[-1] - d[-2])
        # return change in orientation
        # before
        e_b = self.euler_angle_before[2]
        # after
        self.euler_angle = p.getEulerFromQuaternion(p.getLinkState(self.carId, 0)[1])
        e_a = self.euler_angle[2]
        # fix transitions pi <=> -pi
        # in top left quadrant
        e_b_topleft = e_b < np.pi and e_b > np.pi / 2
        e_a_topleft = e_a < np.pi and e_a > np.pi / 2
        # in bottom left quadrant
        e_b_bottomleft = e_b < -np.pi / 2 and e_b > -np.pi
        e_a_bottomleft = e_a < -np.pi / 2 and e_a > -np.pi
        if e_a_topleft and e_b_bottomleft:
            # transition in negative direction
            return -(abs(e_a - np.pi) + abs(e_b + np.pi)),self.agentState,delta_distance
        elif e_a_bottomleft and e_b_topleft:
            # transition in positive direction
            return (abs(e_a + np.pi) + abs(e_b - np.pi)),self.agentState,delta_distance
        else:
            # no transition, just the difference
            return (e_a - e_b),self.agentState,delta_distance


    def change_speed(self, gains):
        p.setJointMotorControlArray(bodyUniqueId=self.carId,
                                    jointIndices=[4, 6],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=gains,
                                    forces=[10, 10])

    def save_position_and_speed(self):
        [position, angle] = p.getBasePositionAndOrientation(self.carId)
        angle = p.getEulerFromQuaternion(angle)
        self.xy_coordinates.append(np.array([position[0], position[1]]))
        self.orientation_angle.append(angle[2])

        [linear_v, _] = p.getBaseVelocity(self.carId)
        self.xy_speeds.append([linear_v[0], linear_v[1]])
        self.speeds.append(np.linalg.norm([linear_v[0], linear_v[1]]))

    def compute_gains(self):
        """Calculates motor gains based on heading and goal vector direction"""
        current_angle = self.orientation_angle[-1]
        current_heading = [np.cos(current_angle), np.sin(current_angle)]
        diff_angle = compute_angle(current_heading, self.goal_vector) / np.pi

        gain = min(np.linalg.norm(self.goal_vector) * 5, 1)

        # If close to the goal do not move
        if gain < 0.5:
            gain = 0

        # If large difference in heading, do an actual turn
        if abs(diff_angle) > 0.05 and gain > 0:
            max_speed = self.max_speed / 2
            direction = np.sign(diff_angle)
            if direction > 0:
                v_left = max_speed * gain * -1
                v_right = max_speed * gain
            else:
                v_left = max_speed * gain
                v_right = max_speed * gain * -1
        else:
            # Otherwise only adjust course slightly
            self.turning = False
            max_speed = self.max_speed
            v_left = max_speed * (1 - diff_angle * 2) * gain
            v_right = max_speed * (1 + diff_angle * 2) * gain

        return [v_left, v_right]

    def end_simulation(self):
        p.disconnect()

    def avoid_obstacles(self, gc_network, pc_network, cognitive_map, exploration_phase,currentHD):
        """Main controller function, to check for obstacles and adjust course if needed."""
        ray_reference = p.getLinkState(self.carId, 0)[1] # return [linkworldposition,linkworldorientation]
        # current_heading = p.getEulerFromQuaternion(ray_reference)[2] if currentHD==None else currentHD
        if currentHD==None:
            current_heading= p.getEulerFromQuaternion(ray_reference)[2]
        else:
            current_heading=currentHD
        # current_heading = p.getEulerFromQuaternion(ray_reference)[2]  # convert into [roll around x,pitch around y,yaw around z]in radians
        goal_vector_angle = np.arctan2(self.goal_vector[1], self.goal_vector[0])
        angles = np.linspace(0, 2 * np.pi, num=self.num_ray_dir, endpoint=False) # resolution of 2pi in num_ray_dir
        # angles = np.linspace(0, 2 * np.pi, num=self.num_ray_dir, endpoint=True)
        # angles = np.linspace(np.pi/2, 2.5 * np.pi, num=self.num_ray_dir, endpoint=False)
        # direction where we want to check for obstacles
        angles = np.append(angles, [goal_vector_angle, current_heading])

        # ray_dist = self.ray_detection( )
        ray_dist = self.ray_detection(angles) # return the distance from agent to hitposition, no information if no hit
        changed = self.update_directions(ray_dist)  # check if an direction became unblocked
        # the ray_detection starts from pi/2 for BC
        # but the angles starts from 0, so creat new angles from pi/2
        # angles = np.linspace(np.pi/2, 2.5 * np.pi, num=self.num_ray_dir, endpoint=False)
        # angles = np.append(angles, [goal_vector_angle, current_heading])
        if np.all(self.directions) and self.topology_based:
            # Switch back to vector-based navigation if all directions are free
            self.topology_based = False
            find_new_goal_vector(gc_network, pc_network, cognitive_map, self,
                                 model="linear_lookahead", pod=None, spike_detector=None)

        minimum_dist = np.min(ray_dist)
        if minimum_dist < 0.3:
            # Initiate back up maneuver
            # give a opposite move direction
            idx = np.argmin(ray_dist)
            angle = angles[idx] + np.pi
            self.goal_vector = np.array([np.cos(angle), np.sin(angle)]) * 0.5
            self.goal_vector_original = self.goal_vector
            self.topology_based = True

        if not exploration_phase:
            if self.topology_based or ray_dist[-1] < 0.6 or ray_dist[-2] < 0.6:
                # Approaching an obstacle in heading or goal vector direction, or topology based
                if not self.topology_based or changed:
                    # Switching to topology-based, or already topology-based but new direction became available
                    self.topology_based = True
                    pick_intermediate_goal_vector(gc_network, pc_network, cognitive_map, self)

        return self.compute_gains()

    def update_directions(self, ray_dist):
        """Check which of the directions are blocked and if one became unblocked"""
        changed = False
        directions = np.ones_like(self.directions) # all 16 directions are true by default
        for idx in range(self.num_ray_dir):
            left = idx - 1 if idx - 1 >= 0 else self.num_ray_dir - 1  # determine left direction in circle
            right = idx + 1 if idx + 1 <= self.num_ray_dir - 1 else 0  # determine right direction in circle
            if ray_dist[idx] < 1.3 or ray_dist[left] < 0.9 or ray_dist[right] < 0.9:
                # If in direction an obstacle is nearby or in one of the neighbouring, then it is blocked
                directions[idx] = False
            if idx % self.num_travel_dir == 0 and directions[idx] and not self.directions[idx]:
                # One of the traveling directions became unblocked
                changed = True
        self.directions = directions # update self.directions
        return changed

    def getPosition(self):
        position = p.getLinkState(self.carId, 0)[0]
        return position

    # def ray_detection(self):
    #     # the index of the ray is from the front, counter-clock-wise direction #
    #     # detect range rayLen = 1 #
    #     p.removeAllUserDebugItems()
    #     rayReturn = []
    #     rayFrom = []
    #     rayTo = []
    #     rayIds = []
    #     numRays=self.num_ray_dir
    #     # numRays = 51
    #     if self.env_model=="maze" or "curved":
    #         # set rayLength in parametersBC
    #         rayLen = parametersBC.rayLength
    #     else:
    #         rayLen = 1
    #     rayHitColor = [1, 0, 0]
    #     rayMissColor = [1, 1, 1]
    #
    #     replaceLines = True
    #
    #     for i in range(numRays):
    #         # rayFromPoint = p.getBasePositionAndOrientation(self.carId)[0]
    #         rayFromPoint = p.getLinkState(self.carId, 0)[0]
    #         rayReference = p.getLinkState(self.carId, 0)[1]
    #         euler_angle = p.getEulerFromQuaternion(rayReference)  # in degree
    #         # print("Euler Angle: ", rayFromPoint)
    #         rayFromPoint = list(rayFromPoint)
    #         rayFromPoint[2] = rayFromPoint[2] + 0.02
    #         rayFrom.append(rayFromPoint)
    #         rayTo.append([
    #             rayLen * math.cos(
    #                 2.0 * math.pi * float(i) / numRays + 360.0 / numRays / 2 * math.pi / 180 + euler_angle[2]) +
    #             rayFromPoint[0],
    #             rayLen * math.sin(
    #                 2.0 * math.pi * float(i) / numRays + 360.0 / numRays / 2 * math.pi / 180 + euler_angle[2]) +
    #             rayFromPoint[1],
    #             rayFromPoint[2]
    #         ])
    #
    #         # if (replaceLines):
    #         #     if i == 0:
    #         #         # rayIds.append(p.addUserDebugLine(rayFrom[i], rayTo[i], [0, 0, 1]))
    #         #         pass
    #         #     else:
    #         #         # rayIds.append(p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor))
    #         #         pass
    #         # else:
    #         #     rayIds.append(-1)
    #
    #     results = p.rayTestBatch(rayFrom, rayTo, numThreads=0)
    #     for i in range(numRays):
    #         hitObjectUid = results[i][0]
    #
    #         if (hitObjectUid < 0):
    #             hitPosition = [0, 0, 0]
    #             # p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor, replaceItemUniqueId=rayIds[i])
    #             if(i ==0):
    #                 p.addUserDebugLine(rayFrom[i], rayTo[i], (0,0,0))
    #             p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor)
    #             rayReturn.append(-1)
    #         else:
    #             hitPosition = results[i][3]
    #             # p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, replaceItemUniqueId=rayIds[i])
    #             p.addUserDebugLine(rayFrom[i], rayTo[i], rayHitColor)
    #             rayReturn.append(
    #                 math.sqrt((hitPosition[0] - rayFrom[i][0]) ** 2 + (hitPosition[1] - rayFrom[i][1]) ** 2))
    #
    #     # self.euler_angle = euler_angle
    #     # print("euler_angle: ", euler_angle[2] * 180 / np.pi)
    #
    #     ### BC-Model
    #     # returns the distance to the walls hit starting from 0 to 2pi counter clockwise so each of the 51 entries is
    #     # the length for one radial separation bin
    #     self.raysThatHit = rayReturn
    #     ###
    #     return rayReturn



    def ray_detection(self, angles):
        """Check for obstacles in defined directions,return distance from agent to hitposition"""
        # for BC Model, the number of rays is better to use uneven number, so the plotmap will be symmetry
        # for Navigation, the number of rays has to be multiples of 4,[W,N,S,E]
        # if we use e.g.17 rays for BC Model, and 16 for Navigation, will detect twice at each step
        # that will need too much time or computer power, it is a trade of between accurate and time comsuming
        # this work is combination of all Cells and need lots of computer power
        # so all the necessary methode will be carried out to reduce computer power or time consuming
        p.removeAllUserDebugItems() # remove all debug items(text,lines etc)

        # ray_len = 2  # max_ray length to check for
        if self.env_model=="maze" or "curved":
            # set rayLength in parametersBC
            ray_len = parametersBC.rayLength
        else:
            ray_len = 2
        rayFrom = []
        rayTo = []
        rayReturn=[]
        ray_from_point = np.array(p.getLinkState(self.carId, 0)[0]) # current position[x,y,z]
        ray_from_point[2] = ray_from_point[2] + 0.02 # z+0,2
        rayReference = p.getLinkState(self.carId, 0)[1]
        euler_angle = p.getEulerFromQuaternion(rayReference)
        rayHitColor = [1, 0, 0]
        rayMissColor = [0, 1, 0]
        #
        # for idx,angle in enumerate(angles):
        #     rayFromPoint = p.getLinkState(self.carId, 0)[0]
        #     rayReference = p.getLinkState(self.carId, 0)[1]
        #     euler_angle = p.getEulerFromQuaternion(rayReference)  # in degree
        #     # print("Euler Angle: ", rayFromPoint)
        #     rayFromPoint = list(rayFromPoint)
        #     rayFromPoint[2] = rayFromPoint[2] + 0.02
        #     rayFrom.append(rayFromPoint)
        #     rayTo.append([
        #         ray_len * math.cos(
        #             2.0 * math.pi * float(idx) / self.num_ray_dir + 360.0 / self.num_ray_dir / 2 * math.pi / 180 + euler_angle[2]) +
        #         rayFromPoint[0],
        #         ray_len * math.sin(
        #             2.0 * math.pi * float(idx) / self.num_ray_dir + 360.0 / self.num_ray_dir / 2 * math.pi / 180 + euler_angle[2]) +
        #         rayFromPoint[1],
        #         rayFromPoint[2]
        #     ])
        for angle in angles:
        #calculate the end position of 18 ray directions, when lenght of ray is 2,in allocentric frame
            rayFrom.append(ray_from_point)
            rayTo.append(np.array([
                # np.cos(angle+ 360.0 / self.num_ray_dir / 2 * np.pi / 180 + euler_angle[2]) * ray_len + ray_from_point[0],
                # np.sin(angle+ 360.0 / self.num_ray_dir / 2 * np.pi / 180 + euler_angle[2]) * ray_len + ray_from_point[1],
                # np.cos(angle + euler_angle[2]) * ray_len + ray_from_point[0], #
                # np.sin(angle + euler_angle[2]) * ray_len + ray_from_point[1], #
                np.cos(angle) * ray_len + ray_from_point[0],
                np.sin(angle) * ray_len + ray_from_point[1],
                ray_from_point[2]
            ]))

        ray_dist = np.empty_like(angles)
        #
        results = p.rayTestBatch(rayFrom, rayTo, numThreads=0)
        # result=[objectUniqueId, lindindex of the hit object,hit fraction,hitposition,hitnormal]
        # result[0]:ObjectId,-1 for not hit,result[3]:hitposition[x,y,z]
        for idx, result in enumerate(results):
            hit_object_uid = result[0]
            # if hit_object_uid==-1, then not hit, the distance from ray_from to ray_to is ray_len=2, overwrite it in ray_dist
            # if hit_object_uid!=-1, then hit, calculate the distance and overwrite it in ray_dist
            dist = ray_len
            # p.addUserDebugLine(ray_from_point, [np.cos(euler_angle),np.sin(euler_angle),ray_from_point[2]], (0, 0, 0))
            if hit_object_uid != -1:
                hit_position = result[3]
                dist = np.linalg.norm(hit_position - ray_from_point)
                ray_dist[idx] = dist
                p.addUserDebugLine(rayFrom[idx], rayTo[idx], rayHitColor)
                rayReturn.append(dist)
                # if (idx==4):
                #     p.addUserDebugLine(rayFrom[idx],rayTo[idx],(0,0,0))
            else:
                # if (idx==4):
                #     p.addUserDebugLine(rayFrom[idx],rayTo[idx],(0,0,0))
                p.addUserDebugLine(rayFrom[idx],rayTo[idx], rayMissColor)
                rayReturn.append(-1)

            ray_dist[idx] = dist # distance from agent to hitposition,if hit,else dist=ray_len2.5
            # ray_dist[idx]=dist if hit, else raylength, need modify according to its use
            # rayReturn[idx]=dist if hit, else -1
            # if dist < 1:
            #     p.addUserDebugLine(ray_from[idx], ray_to[idx], (1, 1, 1))
        index_to_remove=[len(rayReturn)-2,len(rayReturn)-1] #remove the value for Goal_vector and Headdirection
        Rayreturn=np.delete(rayReturn,index_to_remove) #BC Model do not need that 2 values
        self.raysThatHit = Rayreturn # from heading=0, 16 angles,if no hit,-1,else distance to hit position, use for BC
        # ray_dist=[2.5, 2.5,2.5,1.2,..2.5] 16 values
        # rayReturn=[-1,-1,-1,1.2...-1] 16 values
        return ray_dist

    def getRays(self):
        return self.raysThatHit




