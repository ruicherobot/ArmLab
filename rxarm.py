"""!
Implements the RXArm class.

The RXArm class contains:

* last feedback from joints
* functions to command the joints
* functions to get feedback from joints
* functions to do FK and IK
* A function to read the RXArm config file

You will upgrade some functions and also implement others according to the comments given in the code.
"""
import numpy as np
from functools import partial
from kinematics import FK_dh, get_euler_angles_from_T, get_pose_from_T, IK_geometric
import time
import csv
from builtins import super
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
from interbotix_robot_arm import InterbotixRobot
from interbotix_descriptions import interbotix_mr_descriptions as mrd
from config_parse import *
from sensor_msgs.msg import JointState
import rospy

"""
TODO: Implement the missing functions and add anything you need to support them
"""
""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


def _ensure_initialized(func):
    """!
    @brief      Decorator to skip the function if the RXArm is not initialized.

    @param      func  The function to wrap

    @return     The wraped function
    """
    def func_out(self, *args, **kwargs):
        if self.initialized:
            return func(self, *args, **kwargs)
        else:
            print('WARNING: Trying to use the RXArm before initialized')

    return func_out


class RXArm(InterbotixRobot):
    """!
    @brief      This class describes a RXArm wrapper class for the rx200
    """
    def __init__(self, dh_config_file=None):
        """!
        @brief      Constructs a new instance.

                    Starts the RXArm run thread but does not initialize the Joints. Call RXArm.initialize to initialize the
                    Joints.

        @param      dh_config_file  The configuration file that defines the DH parameters for the robot
        """
        super().__init__(robot_name="rx200", mrd=mrd)
        self.joint_names = self.resp.joint_names
        self.num_joints = 5
        # Gripper
        self.gripper_state = 1
        # State
        self.initialized = False
        # Cmd
        self.position_cmd = None
        self.moving_time = 100   ##########original   2.0
        self.accel_time = 5  ######original 0.5
        # Feedback
        self.position_fb = None
        self.velocity_fb = None
        self.effort_fb = None
        # DH Params
        self.dh_params = []
        #self.dh_config_file = dh_config_file
        #if (dh_config_file is not None):
        #    self.dh_params = RXArm.parse_dh_param_file(dh_config_file)

        with open("config/rx200_dh.csv", "r") as file:
            reader = csv.reader(file, delimiter = ',')

            counter = 0
            for row in reader:
                params = []
                for i in range(len(row)):
                    if counter == 0:
                        continue
                    
                    else:
                        params.append(float(row[i]))
                counter += 1
                self.dh_params.append(params)
            self.dh_params = self.dh_params[1:]

            self.dh_params = np.array(self.dh_params)

        #POX params
        self.M_matrix = []
        self.S_list = []

    def initialize(self):
        """!
        @brief      Initializes the RXArm from given configuration file.

                    Initializes the Joints and serial port

        @return     True is succes False otherwise
        """
        self.initialized = False
        # Wait for other threads to finish with the RXArm instead of locking every single call
        rospy.sleep(0.25)
        """ Commanded Values """
        self.position = [0.0] * self.num_joints  # radians
        """ Feedback Values """
        self.position_fb = [0.0] * self.num_joints  # radians
        self.velocity_fb = [0.0] * self.num_joints  # 0 to 1 ???
        self.effort_fb = [0.0] * self.num_joints  # -1 to 1

        # Reset estop and initialized
        self.estop = False
        self.enable_torque()
        self.moving_time = 2.0
        self.accel_time = 0.5
        self.set_gripper_pressure(1.0)
        self.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=False)
        self.open_gripper()
        self.initialized = True
        return self.initialized

    def sleep(self):
        self.moving_time = 2.0
        self.accel_time = 1.0
        self.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=True)
        self.go_to_sleep_pose(moving_time=self.moving_time,
                              accel_time=self.accel_time,
                              blocking=False)
        self.initialized = False

    def set_positions(self, joint_positions):
        """!
         @brief      Sets the positions.

         @param      joint_angles  The joint angles
         """
        self.set_joint_positions(joint_positions,
                                 moving_time=self.moving_time,
                                 accel_time=self.accel_time,
                                 blocking=False)

    def set_moving_time(self, moving_time):
        self.moving_time = moving_time

    def set_accel_time(self, accel_time):
        self.accel_time = accel_time

    def disable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.torque_joints_off(self.joint_names)

    def enable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.torque_joints_on(self.joint_names)

    def get_positions(self):
        """!
        @brief      Gets the positions.

        @return     The positions.
        """
        return self.position_fb

    def get_velocities(self):
        """!
        @brief      Gets the velocities.

        @return     The velocities.
        """
        return self.velocity_fb

    def get_efforts(self):
        """!
        @brief      Gets the loads.

        @return     The loads.
        """
        return self.effort_fb


#   @_ensure_initialized

    def get_ee_pose(self):
        """!
        @brief      TODO Get the EE pose.

        @return     The EE pose as [x, y, z, phi] or as needed.
        """

        angles = self.get_positions()
        H = FK_dh(self.dh_params, angles, 5)
        pose = get_pose_from_T(H)
        return pose

    def test_FK(self, joints):

        ee_pose = self.get_ee_pose()

        # temp1 = IK_geometric1(self.dh_params, ee_pose)
        # temp1 = temp1[0].reshape(1,-1)
        # temp1 = np.append(temp1[0], 0)

        #temp = IK_geometric(self.dh_params, ee_pose)
        # temp = temp2
        '''
        print("---actual angles---")
        gp = self.get_positions()
        print(gp)

        print("--calculated angle--")
        print(temp)

        print("------temp3-gp3------")
        print(temp-gp)

        print("------temp3+gp3------")
        print(temp+gp)

        print("---actual ee pos---")
        print(ee_pose)

        print("-calculated ee pos-")
        print(get_pose_from_T(FK_dh(self.dh_params, joints, 5)))
        '''
    def world_to_joint(self, world_coords):
        # if world_coords[0] > 0:
        #      world_coords[0] += 15
        # else:
        #     world_coords[0] -= 15
        return IK_geometric(self.dh_params, world_coords)

    @_ensure_initialized
    def get_wrist_pose(self):
        """!
        @brief      TODO Get the wrist pose.

        @return     The wrist pose as [x, y, z, phi] or as needed.
        """
        return [0, 0, 0, 0]

    def parse_pox_param_file(self):
        """!
        @brief      TODO Parse a PoX config file

        @return     0 if file was parsed, -1 otherwise 
        """
        return -1

    def parse_dh_param_file(self):
        print("Parsing DH config file...")
        parse_dh_param_file(self.dh_config_file)
        print("DH config file parse exit.")
        return dh_params

    def get_dh_parameters(self):
        """!
        @brief      Gets the dh parameters.

        @return     The dh parameters.
        """
        return self.dh_params


class RXArmThread(QThread):
    """!
    @brief      This class describes a RXArm thread.
    """
    updateJointReadout = pyqtSignal(list)
    updateEndEffectorReadout = pyqtSignal(list)

    def __init__(self, rxarm, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      RXArm  The RXArm
        @param      parent  The parent
        @details    TODO: set any additional initial parameters (like PID gains) here
        """

        
        QThread.__init__(self, parent=parent)
        self.rxarm = rxarm
        '''
        self.pid_gains={rxarm.joint_names[0]:[2000,100,4500],\
                        rxarm.joint_names[1]:[2000,10,500],\
                        rxarm.joint_names[2]:[2000,100,500],\
                        rxarm.joint_names[3]:[800,100,500],\
                        rxarm.joint_names[4]:[640,100,3600],\
                        rxarm.joint_names[5]:[640,10,3600]}
        
        self.pid_gains = {self.rxarm.joint_names[0]:[640,0,3600],\
                        self.rxarm.joint_names[1]:[800,0,0],\
                        self.rxarm.joint_names[2]:[800,0,0],\
                        self.rxarm.joint_names[3]:[800,0,0],\
                        self.rxarm.joint_names[4]:[640,0,3600],\
                        self.rxarm.joint_names[5]:[640,0,3600]}
        '''
        self.pid_gains={rxarm.joint_names[0]:[5000,0,9000],\
                        rxarm.joint_names[1]:[5000,0,500],\
                        rxarm.joint_names[2]:[2000,300,0],\
                        rxarm.joint_names[3]:[800,100,0],\
                        rxarm.joint_names[4]:[640,100,3600],\
                        rxarm.joint_names[5]:[640,10,3600]}
        for joint_names in self.pid_gains.keys():
            self.rxarm.set_joint_position_pid_params(joint_names, self.pid_gains[joint_names])


        rospy.Subscriber('/rx200/joint_states', JointState, self.callback)
        rospy.sleep(0.5)

    def callback(self, data):
        self.rxarm.position_fb = np.asarray(data.position)[0:5]
        self.rxarm.velocity_fb = np.asarray(data.velocity)[0:5]
        self.rxarm.effort_fb = np.asarray(data.effort)[0:5]
        self.updateJointReadout.emit(self.rxarm.position_fb.tolist())
        self.updateEndEffectorReadout.emit(self.rxarm.get_ee_pose())
        #for name in self.rxarm.joint_names:
        #    print("{0} gains: {1}".format(name, self.rxarm.get_motor_pid_params(name)))
        if (__name__ == '__main__'):
            print(self.rxarm.position_fb)

    def run(self):
        """!
        @brief      Updates the RXArm Joints at a set rate if the RXArm is initialized.
        """
        while True:

            rospy.spin()


if __name__ == '__main__':
    rxarm = RXArm()
    print(rxarm.joint_names)
    armThread = RXArmThread(rxarm)
    armThread.start()
    try:
        joint_positions = [-1.0, 0.5, 0.5, 0, 1.57]
        rxarm.initialize()

        rxarm.go_to_home_pose()
        rxarm.set_gripper_pressure(0.5)
        rxarm.set_joint_positions(joint_positions,
                                  moving_time=2.0,
                                  accel_time=0.5,
                                  blocking=True)
        rxarm.close_gripper()
        rxarm.go_to_home_pose()
        rxarm.open_gripper()
        rxarm.sleep()

    except KeyboardInterrupt:
        print("Shutting down")
