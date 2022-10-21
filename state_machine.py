"""!
The state machine that implements the logic.
"""
from re import S
from cv2 import threshold

from numpy import block
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
from kinematics import FK_dh, clamp, IK_geometric
import rospy
from rxarm import RXArm
import csv
import cv2
from pyquaternion import Quaternion
from camera import Camera

D2R = np.pi / 180.0
R2D = 180.0 / np.pi

def calculate_direction(angle, w, h):
    if h > w:
        return -angle
    else:
        return -angle + 90


class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = []
        self.gripperState = [] # 1 open / 0 closed
        #added
        self.H = np.array([[1, 0, 0, 24],[0, -1, 0, 29 - 20 + 200],[0, 0, -1, 978],[0, 0, 0, 1]])
        
              
    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "recordWaypoint":
            self.recordWaypoint()

        if self.next_state == "loadWaypoint":
            self.loadWaypoints()

        if self.next_state == "saveWaypoint":
            self.saveWaypoints()

        if self.next_state == "openGripper":
            self.openGripper()

        if self.next_state == "closeGripper":
            self.closeGripper()

        if self.next_state == "openState":
            self.openState()

        if self.next_state == "closeState":
            self.closeState()


    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.status_message = "State: Execute - Executing motion plan"
        self.next_state = "idle"
        
        rospy.sleep(2)

        for i in range(len(self.waypoints)):
            RXArm.set_positions(self.rxarm, self.waypoints[i])
            
            if(self.gripperState[i]):
                #open
                self.rxarm.open_gripper()
            else:
                #close
                self.rxarm.close_gripper()
            
            rospy.sleep(2)

        self.next_state = "idle"

    def openState(self):
        self.current_state = "openState"
        self.rxarm.gripper_state = 1
        print("OPEN_STATE")
        self.next_state = "idle"

    def closeState(self):
        self.current_state = "closeState"
        self.rxarm.gripper_state = 0
        print("CLOSE_STATE")
        self.next_state = "idle"

    def openGripper(self):
        self.current_state = "openGripper"
        self.rxarm.open_gripper()
        self.rxarm.gripper_state = 1
        self.next_state = "idle"

    def closeGripper(self):
        self.current_state = "closeGripper"
        self.rxarm.close_gripper()
        self.rxarm.gripper_state = 0
        self.next_state = "idle"

    def recordWaypoint(self):
        self.current_state = "recordWaypoint"
        self.waypoints.append(self.rxarm.get_positions())
        self.gripperState.append(self.rxarm.gripper_state)
        self.next_state = "idle"

    def loadWaypoints(self):
        self.waypoints = []
        self.gripperState = []
        self.current_state = "loadWaypoint"
        with open("teach_repeat_4.csv", "r") as file:
            reader = csv.reader(file, delimiter = ',')
            for row in reader:
                for i in range(len(row)):
                    row[i] = float(row[i])
                self.waypoints.append(row[:-1])
                self.gripperState.append(row[-1])
        self.next_state = "idle"

    def saveWaypoints(self):
        self.current_state = "saveWaypoint"
        with open("teach_repeat_4.csv", "wb") as file:
            writer = csv.writer(file)
            for i in range(len(self.waypoints)):
                writer.writerow(np.hstack((self.waypoints[i], self.gripperState[i])))
                
        self.next_state = "idle"

    def clear_waypoints(self):
        self.waypoints = []
        self.gripperState = []
        
    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"
        tag_position_c = np.zeros((4,3))
        print('~~~~~~~~~~~~Start Calibration~~~~~~~~~~~~~')
        if len(self.camera.tag_detections.detections) < 4:
            self.status_message = "No enough 4 tags"
        else:
            for i in range(4):
                tag_position_c[self.camera.tag_detections.detections[i].id[0] - 1,0] = self.camera.tag_detections.detections[i].pose.pose.pose.position.x
                tag_position_c[self.camera.tag_detections.detections[i].id[0] - 1,1] = self.camera.tag_detections.detections[i].pose.pose.pose.position.y
                tag_position_c[self.camera.tag_detections.detections[i].id[0] - 1,2] = self.camera.tag_detections.detections[i].pose.pose.pose.position.z

        tag_position_c = np.transpose(tag_position_c).astype(np.float32)

        # scale tag_position_c ?
        for i in range(4):
            tag_position_c[:, i] /=  tag_position_c[2,i]

        tag_position_i = np.dot(self.camera.intrinsic_matrix,tag_position_c).astype(np.float32)
        
        self.status_message = "Calibration - Completed Calibration"

        (success, rot_vec, trans_vec) = cv2.solvePnP(self.camera.tag_locations.astype(np.float32), np.transpose(tag_position_i[:2, :]).astype(np.float32), self.camera.intrinsic_matrix,self.camera.dist_coefficient, flags = cv2.SOLVEPNP_ITERATIVE)

        dst = cv2.Rodrigues(rot_vec)
        dst = np.array(dst[0])
        
        trans_vec = np.squeeze(trans_vec)
        self.camera.extrinsic_matrix[:3, :3] = dst
        self.camera.extrinsic_matrix[:3, 3] = trans_vec

        # Tilted plane fix
        uv_coords = tag_position_i.astype(int)
        intrinsic_inv = np.linalg.inv(self.camera.intrinsic_matrix)
        camera_coords =  np.matmul(intrinsic_inv, uv_coords)

        for i in range(4):
            tag_position_c[:, i] /=  tag_position_c[2,i]
            z = self.camera.DepthFrameRaw[uv_coords[1,i]][uv_coords[0,i]]
            camera_coords[:,i] *= z

        camera_coords = np.append(camera_coords, [[float(1),float(1),float(1),float(1)]], axis=0)
        world_coords = np.matmul(np.linalg.inv(self.camera.extrinsic_matrix), camera_coords)
        
        # Cross product of tilted frame
        id1 = 1
        id2 = 2
        cross_w = np.cross(world_coords[:3,id1], world_coords[:3,id2])
        b=np.linalg.norm(cross_w)
        cross_w = cross_w / b

        w_points = np.append(np.expand_dims(world_coords[:3,id1], axis = 1),np.expand_dims(world_coords[:3,id2], axis = 1), axis = 1)
        w_points = np.append(w_points,np.expand_dims(cross_w, axis = 1), axis = 1)

        # Cross product of true locations
        true_locations = np.transpose(self.camera.tag_locations)
        cross_t = np.cross(true_locations[:,id1], true_locations[:,id2])
        t=np.linalg.norm(cross_t)
        cross_t = cross_t / t

        t_points = np.append(np.expand_dims(true_locations[:,id1], axis = 1),np.expand_dims(true_locations[:,id2], axis = 1), axis = 1)
        t_points = np.append(t_points,np.expand_dims(cross_t, axis = 1), axis = 1)

        # Cross product for rotation axis
        rot_axis = np.cross(cross_w, cross_t)
        mag_rot=np.linalg.norm(rot_axis)
        rot_axis = rot_axis / mag_rot

        # Angle of rotation
        dot_product = np.dot(cross_w, cross_t)
        angle = -np.arccos(dot_product)/2

        # Quaternion rotation around the axis
        q_rotation = Quaternion(axis = rot_axis, angle = angle)

        # From Quaternion to rotation
        #R = q_rotation.transformation_matrix
        R = np.array([[1, 0, 0, 0],[0 ,np.math.cos(angle), - np.math.sin(angle), 0],[0, np.math.sin(angle), np.math.cos(angle), 10], [0, 0, 0, 1]])

        self.camera.extrinsic_matrix = np.matmul(self.camera.extrinsic_matrix, np.linalg.inv(R))
        
        print("Extrinsic Matrix: " + str(self.camera.extrinsic_matrix))

        self.camera.processDepthFrame()

        self.status_message = "Calibration - Completed Calibration"

    def click_pick(self):
        # click
        x = self.camera.last_click[0]
        y = self.camera.last_click[1]
        z = self.camera.DepthFrameRaw[y][x]
        # from click get world coordinates

        world_coordinate = self.camera.uvd2world(x, y, z)
        self.pick("big", world_coordinate, 0)

    def pick(self, block_size, world_coordinate, block_theta, height):
        '''
        # k_move=1.0, k_accel=1.0/10, min_move_time=0.5, sleep_time=0.1
        k_move=1  # moving_time = k_move * max_angle_bias
        k_accel=0.2  # accel_time = k_accel * moving_time
        min_move_time=1.0
        sleep_time=0.2
        '''
        k_move=1  # moving_time = k_move * max_angle_bias
        k_accel=0.5  # accel_time = k_accel * moving_time
        min_move_time=2.5
        sleep_time=0.5

        grid_size = 5
        phi_original = 180 - grid_size
        phi_up = phi_original
        phi_down = phi_original
        world_coords_up = np.append(world_coordinate, phi_up)
        world_coords_down = np.append(world_coordinate, phi_down)
        world_coords_up_fixed = world_coordinate[2] + height
        world_coords_up[2] += height

        while (any(np.isnan(self.rxarm.world_to_joint(world_coords_up))) and phi_up >= grid_size):
            phi_up -= grid_size
            world_coords_up[3] = phi_up
            world_coords_up[2] = world_coords_up_fixed

        while (any(np.isnan(self.rxarm.world_to_joint(world_coords_down))) and phi_down >= grid_size):
            phi_down -= grid_size
            world_coords_down[3] = phi_down

            if world_coords_down[2] < 5:
                world_coords_down[2] += 20

        world_coords_down[2] -= 4
        joint_angles_up = self.rxarm.world_to_joint(world_coords_up)
        
        if phi_down == phi_original:
            block_rot = clamp(D2R * (block_theta + 90 + R2D * (joint_angles_up[0])))
        else:
            block_rot = 0

        joint_angles_up = np.append(joint_angles_up, block_rot)
        self.waypoints.append(joint_angles_up)
        self.gripperState.append(1)
        # down
        joint_angles_down = self.rxarm.world_to_joint(world_coords_down)
        
        joint_angles_down = np.append(joint_angles_down, block_rot)

        self.waypoints.append(joint_angles_down)
        self.gripperState.append(0)

        self.waypoints.append(joint_angles_up)
        self.gripperState.append(0)

        # execute
        self.status_message = "State: Execute - Executing motion plan"
        for i in range(len(self.waypoints)):
            # print("-------------waypoints[i]-------------")
            # print(self.waypoints[i])

            # Get current position
            current_position = self.rxarm.get_positions()
            current_position = [clamp(deg) for deg in current_position]
             # Get next position
            next_position = self.waypoints[i]
            next_position = [clamp(deg) for deg in next_position]
            # Difference between current and next position
            difference = np.absolute(np.subtract(current_position[:-1], next_position[:-1]))
            # Find largest angle displacement
            max_angle_bias = np.amax(difference)
            # moving time is propotional to largest angle displacement
            moving_time = k_move * max_angle_bias
            # set minimum
            if (moving_time < min_move_time):
               moving_time = min_move_time
            # acceleration time is propotional to moving time
            accel_time = k_accel * moving_time
            # Set
            self.rxarm.set_moving_time(moving_time)
            self.rxarm.set_accel_time(accel_time)
            self.rxarm.set_positions(next_position)
            rospy.sleep(moving_time)

            if(self.gripperState[i]):
                self.rxarm.open_gripper()
            else:
                self.rxarm.close_gripper()
            rospy.sleep(sleep_time)

        self.next_state = "idle"
        # clear here in case these waypoints are executed again in the future
        self.clear_waypoints()
    
    
    def click_place(self):
        x = self.camera.last_click[0]
        y = self.camera.last_click[1]
        z = self.camera.DepthFrameRaw[y][x]

        world_coordinate = self.camera.uvd2world(x, y, z)
        self.place("big", world_coordinate, 0)

    def place(self, block_size, world_coordinate, block_theta, height):
        
        k_move=1  # moving_time = k_move * max_angle_bias
        k_accel=0.5  # accel_time = k_accel * moving_time
        min_move_time=2.5
        sleep_time=0.5
        '''
        k_move=1  # moving_time = k_move * max_angle_bias
        k_accel=0.2  # accel_time = k_accel * moving_time
        min_move_time=1.0
        sleep_time=0.2
        '''
        grid_size = 5
        phi_original = 180 - grid_size
        phi_up = phi_original
        phi_down = phi_original
        world_coords_up = np.append(world_coordinate, phi_up)
        world_coords_down = np.append(world_coordinate, phi_down)

        world_coords_up_fixed = world_coordinate[2] + height
        world_coords_up[2] += height

        # up
        while (any(np.isnan(self.rxarm.world_to_joint(world_coords_up))) and phi_up >= grid_size):
            phi_up -= grid_size
            world_coords_up[3] = phi_up
            world_coords_up[2] = world_coords_up_fixed

        while (any(np.isnan(self.rxarm.world_to_joint(world_coords_down))) and phi_down >= grid_size):
            phi_down -= grid_size
            world_coords_down[3] = phi_down

            if world_coords_down[2] < grid_size:
                world_coords_down[2] += grid_size * 4


        # TODO: manually change z for bad calibration in stack
        # np.abs(world_coords_down[0]) > 350 etc
        # world_coords_down[2] = world_coords_down[2] + 10

        if block_size == "small":
                world_coords_down[2] -= 7.5
        world_coords_down[2] += 30.0
        # up
        joint_angles_up = self.rxarm.world_to_joint(world_coords_up)
        if phi_down == phi_original:
            block_rot = clamp(D2R * (block_theta + 90 + R2D * (joint_angles_up[0])))
        else:
            block_rot = 0
        # if block_rot < 0:
        #     block_rot += 180

        joint_angles_up = np.append(joint_angles_up, block_rot)
        self.waypoints.append(joint_angles_up)
        self.gripperState.append(0)

        # down
        joint_angles_down = self.rxarm.world_to_joint(world_coords_down)
        joint_angles_down = np.append(joint_angles_down, block_rot)

        self.waypoints.append(joint_angles_down)
        self.gripperState.append(1)

        self.waypoints.append(joint_angles_up)
        self.gripperState.append(1)

        # execute
        self.status_message = "State: Execute - Executing motion plan"
        for i in range(len(self.waypoints)):
            # Get current position
            current_position = self.rxarm.get_positions()
            current_position = [clamp(deg) for deg in current_position]
             # Get next position
            next_position = self.waypoints[i]
            next_position = [clamp(deg) for deg in next_position]
            # Difference between current and next position
            difference = np.absolute(np.subtract(current_position[:-1], next_position[:-1]))
            # Find largest angle displacement
            max_angle_bias = np.amax(difference)
            # moving time is propotional to largest angle displacement
            moving_time = k_move * max_angle_bias
            # set minimum
            if (moving_time < min_move_time):
               moving_time = min_move_time
            # acceleration time is propotional to moving time
            accel_time = k_accel * moving_time
            # Set
            self.rxarm.set_moving_time(moving_time)
            self.rxarm.set_accel_time(accel_time)
            self.rxarm.set_positions(next_position)
            rospy.sleep(moving_time)

            if(self.gripperState[i]):
                self.rxarm.open_gripper()
            else:
                self.rxarm.close_gripper()
            rospy.sleep(sleep_time)

        self.next_state = "idle"
        # clear here in case these waypoints are executed again in the future
        self.clear_waypoints()

    def placehigh(self, block_size, world_coordinate, block_theta, height):

        k_move=1  # moving_time = k_move * max_angle_bias
        k_accel=0.5  # accel_time = k_accel * moving_time
        min_move_time=2.5
        sleep_time=0.5
        
        grid_size = 5
        phi_original = 100 - grid_size
        phi_up = phi_original
        phi_down = phi_original
        world_coords_up = np.append(world_coordinate, phi_up)
        world_coords_down = np.append(world_coordinate, phi_down)

        world_coords_up_fixed = world_coordinate[2] + height
        world_coords_up[2] += height

        # up
        while (any(np.isnan(self.rxarm.world_to_joint(world_coords_up))) and phi_up >= grid_size):
            phi_up -= grid_size
            world_coords_up[3] = phi_up
            world_coords_up[2] = world_coords_up_fixed

        while (any(np.isnan(self.rxarm.world_to_joint(world_coords_down))) and phi_down >= grid_size):
            phi_down -= grid_size
            world_coords_down[3] = phi_down

            if world_coords_down[2] < grid_size:
                world_coords_down[2] += grid_size * 4


        world_coords_down[2] += 12
        
        # up
        joint_angles_up = self.rxarm.world_to_joint(world_coords_up)
        if phi_down == phi_original:
            block_rot = clamp(D2R * (block_theta + 90 + R2D * (joint_angles_up[0])))
        else:
            block_rot = 0
        # if block_rot < 0:
        #     block_rot += 180

        joint_angles_up = np.append(joint_angles_up, block_rot)
        self.waypoints.append(joint_angles_up)
        self.gripperState.append(0)

        # down
        joint_angles_down = self.rxarm.world_to_joint(world_coords_down)
        joint_angles_down = np.append(joint_angles_down, block_rot)

        self.waypoints.append(joint_angles_down)
        self.gripperState.append(1)

        self.waypoints.append(joint_angles_up)
        self.gripperState.append(1)

        # execute
        self.status_message = "State: Execute - Executing motion plan"
        for i in range(len(self.waypoints)):
            # Get current position
            current_position = self.rxarm.get_positions()
            current_position = [clamp(deg) for deg in current_position]
             # Get next position
            next_position = self.waypoints[i]
            next_position = [clamp(deg) for deg in next_position]
            # Difference between current and next position
            difference = np.absolute(np.subtract(current_position[:-1], next_position[:-1]))
            # Find largest angle displacement
            max_angle_bias = np.amax(difference)
            # moving time is propotional to largest angle displacement
            moving_time = k_move * max_angle_bias
            # set minimum
            if (moving_time < min_move_time):
               moving_time = min_move_time
            # acceleration time is propotional to moving time
            accel_time = k_accel * moving_time
            # Set
            self.rxarm.set_moving_time(moving_time)
            self.rxarm.set_accel_time(accel_time)
            self.rxarm.set_positions(next_position)
            rospy.sleep(moving_time)

            if(self.gripperState[i]):
                self.rxarm.open_gripper()
            else:
                self.rxarm.close_gripper()
            rospy.sleep(sleep_time)

        self.next_state = "idle"
        # clear here in case these waypoints are executed again in the future
        self.clear_waypoints()

    def kick(self, block_size, world_coordinate, block_theta):

        height=150
        k_move=1  # moving_time = k_move * max_angle_bias
        k_accel=0.5  # accel_time = k_accel * moving_time
        min_move_time=2.0
        sleep_time=0.5
        block_width = 38
        if block_size == "small":
            block_width = 25
        
        grid_size = 5
        phi_original = 180 - grid_size
        phi_up_away = phi_original
        phi_down_away = phi_original
        phi_up_close = phi_original
        phi_down_close = phi_original

        pos = np.math.sqrt(world_coordinate[0]**2 + world_coordinate[1]**2)
        temp_theta = np.math.atan2(world_coordinate[1], world_coordinate[0])

        world_coordinate_away = np.copy(world_coordinate)
        world_coordinate_away[0] += block_width * np.math.cos(temp_theta) * 2
        world_coordinate_away[1] += block_width * np.math.sin(temp_theta) * 2
        world_coordinate_close = np.copy(world_coordinate)
        world_coordinate_close[0] -= block_width * np.math.cos(temp_theta) * 2
        world_coordinate_close[1] -= block_width * np.math.sin(temp_theta) * 2

        world_coords_up_away = np.append(world_coordinate_away, phi_up_away)
        world_coords_down_away = np.append(world_coordinate_away, phi_down_away)
        
        world_coords_up_away_fixed = world_coordinate[2] + height
        world_coords_up_away[2] += height


        world_coords_up_close = np.append(world_coordinate_close, phi_up_close)
        world_coords_down_close = np.append(world_coordinate_close, phi_down_close)
        
        world_coords_up_close_fixed = world_coordinate[2] + height
        world_coords_up_close[2] += height

        while (any(np.isnan(self.rxarm.world_to_joint(world_coords_up_away))) and phi_up_away >= grid_size):
            phi_up_away -= grid_size
            world_coords_up_away[3] = phi_up_away
            world_coords_up_away[2] = world_coords_up_away_fixed

        while (any(np.isnan(self.rxarm.world_to_joint(world_coords_down_away))) and phi_down_away >= grid_size):
            phi_down_away -= grid_size
            world_coords_down_away[3] = phi_down_away

            if world_coords_down_away[2] < 5:
                world_coords_down_away[2] += 20

        while (any(np.isnan(self.rxarm.world_to_joint(world_coords_up_close))) and phi_up_close >= grid_size):
            phi_up_close -= grid_size
            world_coords_up_close[3] = phi_up_close
            world_coords_up_close[2] = world_coords_up_close_fixed

        while (any(np.isnan(self.rxarm.world_to_joint(world_coords_down_close))) and phi_down_close >= grid_size):
            phi_down_close -= grid_size
            world_coords_down_close[3] = phi_down_close

            if world_coords_down_close[2] < 5:
                world_coords_down_close[2] += 20

        # away
        # up
        joint_angles_up_away = self.rxarm.world_to_joint(world_coords_up_away)
        
        if phi_down_away == phi_original:
            block_rot = clamp(D2R * (block_theta + 90 + R2D * (joint_angles_up_away[0])))
        else:
            block_rot = 0

        joint_angles_up_away = np.append(joint_angles_up_away, block_rot)
        
        # down
        joint_angles_down_away = self.rxarm.world_to_joint(world_coords_down_away)
        
        joint_angles_down_away = np.append(joint_angles_down_away, block_rot)


        # close
        # up
        joint_angles_up_close = self.rxarm.world_to_joint(world_coords_up_close)
        
        if phi_down_close == phi_original:
            block_rot = clamp(D2R * (block_theta + 90 + R2D * (joint_angles_up_close[0])))
        else:
            block_rot = 0

        joint_angles_up_close = np.append(joint_angles_up_close, block_rot)
        
        # down
        joint_angles_down_close = self.rxarm.world_to_joint(world_coords_down_close)
        
        joint_angles_down_close = np.append(joint_angles_down_close, block_rot)

        if pos > 300:
            self.waypoints.append(joint_angles_up_away)
            self.gripperState.append(0)

            self.waypoints.append(joint_angles_down_away)
            self.gripperState.append(0)

            self.waypoints.append(joint_angles_down_close)
            self.gripperState.append(0)

            self.waypoints.append(joint_angles_up_close)
            self.gripperState.append(0)

        else:
            self.waypoints.append(joint_angles_up_close)
            self.gripperState.append(0)

            self.waypoints.append(joint_angles_down_close)
            self.gripperState.append(0)

            self.waypoints.append(joint_angles_down_away)
            self.gripperState.append(0)

            self.waypoints.append(joint_angles_up_away)
            self.gripperState.append(0)


        # execute
        self.status_message = "State: Execute - Executing motion plan"
        for i in range(len(self.waypoints)):
            # print("-------------waypoints[i]-------------")
            # print(self.waypoints[i])

            # Get current position
            current_position = self.rxarm.get_positions()
            current_position = [clamp(deg) for deg in current_position]
             # Get next position
            next_position = self.waypoints[i]
            next_position = [clamp(deg) for deg in next_position]
            # Difference between current and next position
            difference = np.absolute(np.subtract(current_position[:-1], next_position[:-1]))
            # Find largest angle displacement
            max_angle_bias = np.amax(difference)
            # moving time is propotional to largest angle displacement
            moving_time = k_move * max_angle_bias
            # set minimum
            if (moving_time < min_move_time):
               moving_time = min_move_time
            # acceleration time is propotional to moving time
            accel_time = k_accel * moving_time
            # Set
            self.rxarm.set_moving_time(moving_time)
            self.rxarm.set_accel_time(accel_time)
            self.rxarm.set_positions(next_position)
            rospy.sleep(moving_time)

            if(self.gripperState[i]):
                self.rxarm.open_gripper()
            else:
                self.rxarm.close_gripper()
            rospy.sleep(sleep_time)

        self.next_state = "idle"
        # clear here in case these waypoints are executed again in the future
        self.clear_waypoints()

    def placePush(self, block_size, world_coordinate, block_theta, height):

        height=150
        k_move=1  # moving_time = k_move * max_angle_bias
        k_accel=0.5  # accel_time = k_accel * moving_time
        min_move_time=2.5
        sleep_time=0.5
        block_width = 38
        if block_size == "small":
            block_width = 25
        
        grid_size = 5
        phi_original = 180 - grid_size
        phi_up_away = phi_original
        phi_down_away = phi_original
        phi_up_close = phi_original
        phi_down_close = phi_original

        world_coordinate_away = np.copy(world_coordinate)
        world_coordinate_close = np.copy(world_coordinate)
        
        if block_width == 25:
            world_coordinate_close[0] -= block_width * 1.5
            world_coordinate_close[2] += 25
            world_coordinate_away[2] += 25
        else:
            world_coordinate_close[0] += block_width * 2
            world_coordinate_close[2] += 35
            world_coordinate_away[2] += 35

        world_coords_up_away = np.append(world_coordinate_away, phi_up_away)
        world_coords_down_away = np.append(world_coordinate_away, phi_down_away)
        
        world_coords_up_away_fixed = world_coordinate[2] + height
        world_coords_up_away[2] += height


        world_coords_up_close = np.append(world_coordinate_close, phi_up_close)
        world_coords_down_close = np.append(world_coordinate_close, phi_down_close)
        
        world_coords_up_close_fixed = world_coordinate[2] + height
        world_coords_up_close[2] += height

        while (any(np.isnan(self.rxarm.world_to_joint(world_coords_up_away))) and phi_up_away >= grid_size):
            phi_up_away -= grid_size
            world_coords_up_away[3] = phi_up_away
            world_coords_up_away[2] = world_coords_up_away_fixed

        while (any(np.isnan(self.rxarm.world_to_joint(world_coords_down_away))) and phi_down_away >= grid_size):
            phi_down_away -= grid_size
            world_coords_down_away[3] = phi_down_away

            if world_coords_down_away[2] < 5:
                world_coords_down_away[2] += 20

        while (any(np.isnan(self.rxarm.world_to_joint(world_coords_up_close))) and phi_up_close >= grid_size):
            phi_up_close -= grid_size
            world_coords_up_close[3] = phi_up_close
            world_coords_up_close[2] = world_coords_up_close_fixed

        while (any(np.isnan(self.rxarm.world_to_joint(world_coords_down_close))) and phi_down_close >= grid_size):
            phi_down_close -= grid_size
            world_coords_down_close[3] = phi_down_close

            if world_coords_down_close[2] < 5:
                world_coords_down_close[2] += 20

        # away
        # up
        joint_angles_up_away = self.rxarm.world_to_joint(world_coords_up_away)
        
        if phi_down_away == phi_original:
            block_rot = clamp(D2R * (block_theta + 90 + R2D * (joint_angles_up_away[0])))
        else:
            block_rot = 0

        joint_angles_up_away = np.append(joint_angles_up_away, block_rot)
        
        # down
        joint_angles_down_away = self.rxarm.world_to_joint(world_coords_down_away)
        
        joint_angles_down_away = np.append(joint_angles_down_away, block_rot)


        # close
        # up
        joint_angles_up_close = self.rxarm.world_to_joint(world_coords_up_close)
        
        if phi_down_close == phi_original:
            block_rot = clamp(D2R * (block_theta + 90 + R2D * (joint_angles_up_close[0])))
        else:
            block_rot = 0

        joint_angles_up_close = np.append(joint_angles_up_close, block_rot)
        
        # down
        joint_angles_down_close = self.rxarm.world_to_joint(world_coords_down_close)
        
        joint_angles_down_close = np.append(joint_angles_down_close, block_rot)

        self.waypoints.append(joint_angles_up_away)
        self.gripperState.append(0)


        self.waypoints.append(joint_angles_down_away)
        self.gripperState.append(0)

        self.waypoints.append(joint_angles_down_close)
        self.gripperState.append(1)

        self.waypoints.append(joint_angles_up_close)
        self.gripperState.append(1)


        # execute
        self.status_message = "State: Execute - Executing motion plan"
        for i in range(len(self.waypoints)):
            # print("-------------waypoints[i]-------------")
            # print(self.waypoints[i])

            # Get current position
            current_position = self.rxarm.get_positions()
            current_position = [clamp(deg) for deg in current_position]
             # Get next position
            next_position = self.waypoints[i]
            next_position = [clamp(deg) for deg in next_position]
            # Difference between current and next position
            difference = np.absolute(np.subtract(current_position[:-1], next_position[:-1]))
            # Find largest angle displacement
            max_angle_bias = np.amax(difference)
            # moving time is propotional to largest angle displacement
            moving_time = k_move * max_angle_bias
            # set minimum
            if (moving_time < min_move_time):
               moving_time = min_move_time
            # acceleration time is propotional to moving time
            accel_time = k_accel * moving_time
            # Set
            self.rxarm.set_moving_time(moving_time)
            self.rxarm.set_accel_time(accel_time)
            self.rxarm.set_positions(next_position)
            rospy.sleep(moving_time)

            if(self.gripperState[i]):
                self.rxarm.open_gripper()
            else:
                self.rxarm.close_gripper()
            rospy.sleep(sleep_time)

        self.next_state = "idle"
        # clear here in case these waypoints are executed again in the future
        self.clear_waypoints()



    def seperate(self, block_size, world_coordinate, block_theta, height, w, h):

        height=150
        k_move=1  # moving_time = k_move * max_angle_bias
        k_accel=0.5  # accel_time = k_accel * moving_time
        min_move_time=2.5
        sleep_time=0.5
        block_width = 38
        if block_size == "small":
            block_width = 25

        temp_theta = calculate_direction(block_theta, w, h)
        
        grid_size = 5
        phi_original = 180 - grid_size
        phi_up_away = phi_original
        phi_down_away = phi_original
        phi_up_close = phi_original
        phi_down_close = phi_original

        world_coordinate_away = np.copy(world_coordinate)
        world_coordinate_close = np.copy(world_coordinate)
        
        print("Hitting angle: ", temp_theta)

        temp_theta = clamp(D2R * temp_theta)

        world_coordinate_away = np.copy(world_coordinate)
        world_coordinate_away[0] += block_width * np.math.cos(temp_theta) * 2
        world_coordinate_away[1] += block_width * np.math.sin(temp_theta) * 2
        world_coordinate_away[2] += 10

        world_coordinate_close = np.copy(world_coordinate)
        world_coordinate_close[0] -= block_width * np.math.cos(temp_theta) * 2
        world_coordinate_close[1] -= block_width * np.math.sin(temp_theta) * 2
        world_coordinate_close[2] += 10

        world_coords_up_away = np.append(world_coordinate_away, phi_up_away)
        world_coords_down_away = np.append(world_coordinate_away, phi_down_away)
        
        world_coords_up_away_fixed = world_coordinate[2] + height
        world_coords_up_away[2] += height


        world_coords_up_close = np.append(world_coordinate_close, phi_up_close)
        world_coords_down_close = np.append(world_coordinate_close, phi_down_close)
        
        world_coords_up_close_fixed = world_coordinate[2] + height
        world_coords_up_close[2] += height

        while (any(np.isnan(self.rxarm.world_to_joint(world_coords_up_away))) and phi_up_away >= grid_size):
            phi_up_away -= grid_size
            world_coords_up_away[3] = phi_up_away
            world_coords_up_away[2] = world_coords_up_away_fixed

        while (any(np.isnan(self.rxarm.world_to_joint(world_coords_down_away))) and phi_down_away >= grid_size):
            phi_down_away -= grid_size
            world_coords_down_away[3] = phi_down_away

            if world_coords_down_away[2] < 5:
                world_coords_down_away[2] += 20

        while (any(np.isnan(self.rxarm.world_to_joint(world_coords_up_close))) and phi_up_close >= grid_size):
            phi_up_close -= grid_size
            world_coords_up_close[3] = phi_up_close
            world_coords_up_close[2] = world_coords_up_close_fixed

        while (any(np.isnan(self.rxarm.world_to_joint(world_coords_down_close))) and phi_down_close >= grid_size):
            phi_down_close -= grid_size
            world_coords_down_close[3] = phi_down_close

            if world_coords_down_close[2] < 5:
                world_coords_down_close[2] += 20

        # away
        # up
        joint_angles_up_away = self.rxarm.world_to_joint(world_coords_up_away)
        
        if phi_down_away == phi_original:
            block_rot = clamp(D2R * (block_theta + 90 + R2D * (joint_angles_up_away[0])))
        else:
            block_rot = 0

        joint_angles_up_away = np.append(joint_angles_up_away, block_rot)
        
        # down
        joint_angles_down_away = self.rxarm.world_to_joint(world_coords_down_away)
        
        joint_angles_down_away = np.append(joint_angles_down_away, -np.math.pi/2)


        # close
        # up
        joint_angles_up_close = self.rxarm.world_to_joint(world_coords_up_close)
        
        if phi_down_close == phi_original:
            block_rot = clamp(D2R * (block_theta + 90 + R2D * (joint_angles_up_close[0])))
        else:
            block_rot = 0

        joint_angles_up_close = np.append(joint_angles_up_close, block_rot)
        
        # down
        joint_angles_down_close = self.rxarm.world_to_joint(world_coords_down_close)
        
        joint_angles_down_close = np.append(joint_angles_down_close, np.math.pi/2)

        self.waypoints.append(joint_angles_up_away)
        self.gripperState.append(0)


        self.waypoints.append(joint_angles_down_away)
        self.gripperState.append(0)

        self.waypoints.append(joint_angles_down_close)
        self.gripperState.append(1)

        self.waypoints.append(joint_angles_up_close)
        self.gripperState.append(1)


        # execute
        self.status_message = "State: Execute - Executing motion plan"
        for i in range(len(self.waypoints)):
            # print("-------------waypoints[i]-------------")
            # print(self.waypoints[i])

            # Get current position
            current_position = self.rxarm.get_positions()
            current_position = [clamp(deg) for deg in current_position]
             # Get next position
            next_position = self.waypoints[i]
            next_position = [clamp(deg) for deg in next_position]
            # Difference between current and next position
            difference = np.absolute(np.subtract(current_position[:-1], next_position[:-1]))
            # Find largest angle displacement
            max_angle_bias = np.amax(difference)
            # moving time is propotional to largest angle displacement
            moving_time = k_move * max_angle_bias
            # set minimum
            if (moving_time < min_move_time):
               moving_time = min_move_time
            # acceleration time is propotional to moving time
            accel_time = k_accel * moving_time
            # Set
            self.rxarm.set_moving_time(moving_time)
            self.rxarm.set_accel_time(accel_time)
            self.rxarm.set_positions(next_position)
            rospy.sleep(moving_time)

            if(self.gripperState[i]):
                self.rxarm.open_gripper()
            else:
                self.rxarm.close_gripper()
            rospy.sleep(sleep_time)

        self.next_state = "idle"
        # clear here in case these waypoints are executed again in the future
        self.clear_waypoints()


    def detect(self):
        """!
        @brief      Detect the blocks
        """
        self.status_message = "State: Detecting blablabla"
        rospy.sleep(1)
        self.next_state = "idle"

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            rospy.sleep(5)
        self.next_state = "idle"
    
    def pick_and_sort(self):
        """!
        @brief      Competition 1
        """
        # Image coordinates
        destination_left = [[300,-135,0],[250,-135,0],[200,-135,0],[150,-135,0],[100,-135,0],[300,-85,0],[250,-85,0],[200,-85,0],[150,-85,0]]
        destination_right = [[-300,-135,0],[-250,-135,0],[-200,-135,0],[-150,-135,0],[-100,-135,0],[-300,-85,0],[-250,-85,0],[-200,-85,0],[-150,-85,0]]

        self.camera.blockDetector()

        i = 0
        j = 0
        thresh = 8

        while len(self.camera.block_detections) > 0:
            for block in self.camera.block_detections:
                world_coords = block.coord

                self.pick(block.size, world_coords, block.theta, 150)
                if(block.size == "small"):
                    self.place(block.size, destination_right[i], 0, 90)
                    i += 1
                else:
                    self.place(block.size, destination_left[j], 0, 90)
                    j += 1
                

            self.rxarm.sleep()
            rospy.sleep(2)
            self.camera.blockDetector()
            rospy.sleep(2)

        self.camera.mask_list = []
    
    def pick_and_stack(self):
        """!
        @brief      Competition 2
        """
        
        destination_bases = [[150,-75,0],[-150,-75,0],[250,-75,0]]

        self.camera.blockDetector()

        i = 0

        block_detect_sorted = []
             

        max_height = 0

        thresh = 6
        flag = 0
        while len(self.camera.block_detections) > 0:
            if (flag == 0):
                count1 = 0
                for block in self.camera.block_detections:
                    w_coords = block.coord
                    if(block.coord[2] > 45):
                        count1 = count1 + 1
                        self.kick(block.size, w_coords, block.theta)
                if (count1 == 0):
                    flag = 1

            if (flag == 1):
                count2 = 0
                for block in self.camera.block_detections:
                    w_coords = block.coord
                    if (np.abs(block.width - block.height) > thresh):
                        self.seperate(block.size, w_coords, block.theta, 150, block.width, block.height)
                        count2 += 1
                if (count2 == 0):
                    flag = 2

            if (flag == 2):
                for block in self.camera.block_detections:
                    if(block.size == "big"):
                        block_detect_sorted.append(block)
                for block in self.camera.block_detections:
                    if(block.size == "small"):
                        block_detect_sorted.append(block)

                for block in block_detect_sorted:
                    w_coords = block.coord
                    self.pick(block.size, w_coords, block.theta, 150)
                    self.place(block.size, destination_bases[i%3], 0, max_height+90)
                    if (block.size == 'big'):
                        destination_bases[i%3][2] += (38 +1)
                    elif (block.size == 'small'):
                        destination_bases[i%3][2] += (25 + 1)
                    if(i%3 == 0):
                        if (block.size == 'big'):
                            max_height +=  (38 + 10)
                        elif (block.size == 'small'):
                            max_height += (25 + 10)
                    i += 1

                block_detect_sorted = []

            self.rxarm.sleep()
            rospy.sleep(2)
            self.camera.blockDetector()
            rospy.sleep(2)

        self.camera.mask_list = []
    
    def line_them_up(self):
        """!
        @brief      Competition 3
        """
        # Image coordinates
        destination_right = [[150,-85,0],[170,-85,0], [190,-85,0], [210,-85,0], [230,-85,0], [250,-85,0]]
        destination_left = [[-150,-85,0],[-190,-85,0], [-230,-85,0], [-270,-85,0], [-310,-85,0], [-350,-85,0]]

        self.camera.blockDetector()

        i = 0
        j = 0

        block_detect_sorted = []

        count = 0
        thresh = 6
        while len(self.camera.block_detections) > 0:
            flag = 0

            for block in self.camera.block_detections:
                w_coords = block.coord
                if(block.coord[2] > 45):
                    flag = 1
                    self.kick(block.size, w_coords, block.theta)

                if (np.abs(block.width - block.height) > thresh and flag != 1):
                    flag = 2
                    self.seperate(block.size, w_coords, block.theta, 150, block.width, block.height)
                    count += 1

                if flag == 0 and count != 0:
                    for block in self.camera.block_detections:
                        if(block.color == "red"):
                            block_detect_sorted.append(block)
                    for block in self.camera.block_detections:
                        if(block.color == "orange"):
                            block_detect_sorted.append(block)
                    for block in self.camera.block_detections:
                        if(block.color == "yellow"):
                            block_detect_sorted.append(block)
                    for block in self.camera.block_detections:
                        if(block.color == "green"):
                            block_detect_sorted.append(block)
                    for block in self.camera.block_detections:
                        if(block.color == "blue"):
                            block_detect_sorted.append(block)
                    for block in self.camera.block_detections:
                        if(block.color == "violet"):
                            block_detect_sorted.append(block)

                    for block in block_detect_sorted:
                        w_coords = block.coord
                        self.pick(block.size, w_coords, block.theta, 150)
                        if(block.size == "small"):
                            self.placePush(block.size, destination_right[i], 0, 90)
                            i += 1
                        else:
                            self.placePush(block.size, destination_left[j], 0, 90)
                            j += 1

                    block_detect_sorted = []
                    break

            '''if flag == 1:
                count = 0
                for block in self.camera.block_detections:
                    if(block.coord[2] > 45):
                        w_coords = block.coord
                        self.kick(block.size, w_coords, block.theta)
                        count += 1
            
            if flag == 0:
                
                for block in self.camera.block_detections:
                    if(block.color == "red"):
                        block_detect_sorted.append(block)
                for block in self.camera.block_detections:
                    if(block.color == "orange"):
                        block_detect_sorted.append(block)
                for block in self.camera.block_detections:
                    if(block.color == "yellow"):
                        block_detect_sorted.append(block)
                for block in self.camera.block_detections:
                    if(block.color == "green"):
                        block_detect_sorted.append(block)
                for block in self.camera.block_detections:
                    if(block.color == "blue"):
                        block_detect_sorted.append(block)
                for block in self.camera.block_detections:
                    if(block.color == "violet"):
                        block_detect_sorted.append(block)

                for block in block_detect_sorted:
                    w_coords = block.coord
                    self.pick(block.size, w_coords, block.theta, 150)
                    if(block.size == "small"):
                        self.placePush(block.size, destination_right[i], 0, 90)
                        i += 1
                    else:
                        self.placePush(block.size, destination_left[j], 0, 90)
                        j += 1

                block_detect_sorted = []
            
            if count == 0:
                flag = 0'''

            self.rxarm.sleep()
            rospy.sleep(2)
            self.camera.blockDetector()
            rospy.sleep(2)
            self.camera.mask_list = []

        '''
        small = 25
        big = 38
        stack_heights = [
            [small,big], #s,b
            [2*small,small+big,2*big], #ss,bs,bb
            [3*small,2*small+big,small+2*big,3*big], #sss,bss,bbs,bbb
            [4*small,3*small+big,2*small+2*big,small+3*big,4*big] #ssss,bsss,bbss,bbbs,bbbb
        ]

        '''
        
    
    def stack_them_high(self):
        """!
        @brief      Competition 4
        """
        destination_right = [150,-75,0]
        destination_left = [-150,-75,0]

        self.camera.blockDetector()

        block_detect_sorted = []

        max_height = 0

        thresh = 6
        flag = 0
        while len(self.camera.block_detections) > 0:
            if (flag == 0):
                count1 = 0
                for block in self.camera.block_detections:
                    w_coords = block.coord
                    if(block.coord[2] > 45):
                        count1 = count1 + 1
                        self.kick(block.size, w_coords, block.theta)
                if (count1 == 0):
                    flag = 1

            if (flag == 1):
                count2 = 0
                for block in self.camera.block_detections:
                    w_coords = block.coord
                    if (np.abs(block.width - block.height) > thresh):
                        self.seperate(block.size, w_coords, block.theta, 150, block.width, block.height)
                        count2 += 1
                if (count2 == 0):
                    flag = 2

            if (flag == 2):
                for block in self.camera.block_detections:
                    if(block.color == "red"):
                        block_detect_sorted.append(block)
                for block in self.camera.block_detections:
                    if(block.color == "orange"):
                        block_detect_sorted.append(block)
                for block in self.camera.block_detections:
                    if(block.color == "yellow"):
                        block_detect_sorted.append(block)
                for block in self.camera.block_detections:
                    if(block.color == "green"):
                        block_detect_sorted.append(block)
                for block in self.camera.block_detections:
                    if(block.color == "blue"):
                        block_detect_sorted.append(block)
                for block in self.camera.block_detections:
                    if(block.color == "violet"):
                        block_detect_sorted.append(block)

                for block in block_detect_sorted:
                        w_coords = block.coord
                        self.pick(block.size, w_coords, block.theta, 150)
                        
                        if (block.size == 'big'):
                            self.place(block.size, destination_right, 0, max_height+100)
                            destination_right[2] += 38
                            max_height +=  32
                        elif (block.size == 'small'):
                            self.place(block.size, destination_left, 0, max_height+100)
                            destination_left[2] += 25
                            max_height += 20

                block_detect_sorted = []

            self.rxarm.sleep()
            rospy.sleep(2)
            self.camera.blockDetector()
            rospy.sleep(2)

            self.camera.mask_list = []
    
    def to_the_sky(self):
        """!
        @brief      Competition 5
        """
        destination = [240,0,0]

        self.camera.blockDetector()


        max_height = 0

        flag = 1
        count = 0
        thresh = 6
        while len(self.camera.block_detections) > 0:
            '''
            flag = 0

            for block in self.camera.block_detections:
                w_coords = block.coord
                if(block.coord[2] > 45):
                    flag = 1
                    self.kick(block.size, w_coords, block.theta)

                if (np.abs(block.width - block.height) > thresh and flag != 1):
                    flag = 2
                    self.seperate(block.size, w_coords, block.theta, 150, block.width, block.height)
                    count += 1

                if flag == 0 and count != 0:
                    index = 1
                    for block in self.camera.block_detections:
                        if index > 4:
                            #special place
                            w_coords = block.coord
                            self.pick(block.size, w_coords, block.theta, 150)
                            self.placehigh(block.size, destination, 0, 150)
                            destination[2] += (38 +1)
                            max_height +=  (38 + 3)
                        else:
                            w_coords = block.coord
                            self.pick(block.size, w_coords, block.theta, 150)
                            self.place(block.size, destination, 0, 150)
                            destination[2] += (38 +1)
                            max_height +=  (38 + 3)

                        index += 1

                    block_detect_sorted = []
                    break

            '''
            if flag == 1:
                count = 0
                for block in self.camera.block_detections:
                    if(block.coord[2] > 45):
                        w_coords = block.coord
                        self.kick(block.size, w_coords, block.theta)
                        count += 1
            
            if flag == 0:
                index = 1
                for block in self.camera.block_detections:
                    w_coords = block.coord
                    self.pick(block.size, w_coords, block.theta, 150)
                    if index > 6:
                        self.placehigh(block.size, destination, 0, 150)
                    else:
                        self.place(block.size, destination, 0, 150)
                    destination[2] += (38 +1)
                    max_height +=  (38 + 3)
                    index = index +1
            
            if count == 0:
                flag = 0
            

            self.rxarm.sleep()
            rospy.sleep(2)
            self.camera.blockDetector()
            rospy.sleep(2)
            self.camera.mask_list = []
    
    def test_phi(self):

        k_move=1  # moving_time = k_move * max_angle_bias
        k_accel=0.5  # accel_time = k_accel * moving_time
        min_move_time=2.5
        sleep_time=0.5
        phi = 95
        zz = 200
        while zz <= 440:
            world_coords = [300, 0, zz, phi]
            joint_angles = self.rxarm.world_to_joint(world_coords)
            
            joint_angles = np.append(joint_angles, 0)
            self.waypoints.append(joint_angles)
            self.gripperState.append(1)
            zz = zz + 20
        # execute
        self.status_message = "State: Execute - Executing motion plan"
        for i in range(len(self.waypoints)):
            # print("-------------waypoints[i]-------------")
            # print(self.waypoints[i])

            # Get current position
            current_position = self.rxarm.get_positions()
            current_position = [clamp(deg) for deg in current_position]
             # Get next position
            next_position = self.waypoints[i]
            next_position = [clamp(deg) for deg in next_position]
            # Difference between current and next position
            difference = np.absolute(np.subtract(current_position[:-1], next_position[:-1]))
            # Find largest angle displacement
            max_angle_bias = np.amax(difference)
            # moving time is propotional to largest angle displacement
            moving_time = k_move * max_angle_bias
            # set minimum
            if (moving_time < min_move_time):
               moving_time = min_move_time
            # acceleration time is propotional to moving time
            accel_time = k_accel * moving_time
            # Set
            self.rxarm.set_moving_time(moving_time)
            self.rxarm.set_accel_time(accel_time)
            self.rxarm.set_positions(next_position)
            rospy.sleep(moving_time)

            if(self.gripperState[i]):
                self.rxarm.open_gripper()
            else:
                self.rxarm.close_gripper()
            rospy.sleep(sleep_time)

        self.next_state = "idle"
        # clear here in case these waypoints are executed again in the future
        self.clear_waypoints()

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            rospy.sleep(0.05)