"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

from itertools import count
from logging import logMultiprocessing
import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm, logm

D2R = np.pi / 180.0
R2D = 180.0 / np.pi

def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    H = np.eye(4)
    for i in range(link):
        H = np.dot(H, get_transform_from_dh(dh_params[i, 0], dh_params[i, 1], dh_params[i, 2], dh_params[i, 3] + joint_angles[i]))
    return H


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transform matrix.
    """
    return np.matrix([[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
                            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                            [0, np.sin(alpha), np.cos(alpha), d],
                            [0, 0, 0, 1]])


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the Euler angles from a T matrix
ler angles from a transformation matrix.

                TODO: Implement this function return the Euler angles from a T matrix

    @param      T     transformatio
    """
    euler = np.zeros((3, 1))
    R23 = T[1, 2]
    R13 = T[0, 2]
    R33 = T[2, 2]
    R32 = T[2, 1]
    R31 = T[2, 0]
    r = np.arctan2(R23, R13)
    p = np.arctan2(np.math.sqrt(1 - R33 ** 2), R33)
    y = np.arctan2(R32, -R31)
    euler[0, 0] = r
    euler[1, 0] = p
    euler[2, 0] = y

    return euler 

def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the joint pose from a T matrix of the form (x,y,z,phi) where phi is
                rotation about base frame y-axis

    @param      T     transformation matrix

    @return     The pose from T.
    """
    phi = get_euler_angles_from_T(T)[1, 0] * R2D
    pose = T[:, 3:4]
    pose[3, 0] = phi
    return list(pose)

def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a 4-tuple (x, yself.rxarm.get_ee_pose() z, phi) representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4-tuple (x, y, z, phi) representing the pose of the desired link note: phi is the euler
                angle about y in the base frame

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4-tuple (x, y, z, phi) representing the pose of the desired link
    """
    pass


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass


def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.
                TODO: Convert a desired end-effector pose as np.array x,y,z,phi to joint angles
    @param      dh_params  The dh parameters
    @param      pose       The desired pose as np.array x,y,z,phi
    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """

    l2_offset = np.arctan(1.0 / 4.0)

    if type(pose[0])==np.matrix:
        xe = pose[0].item()
    else:
        xe = pose[0]
    if type(pose[1])==np.matrix:
        ye = pose[1].item()
    else:
        ye = pose[1]
    if type(pose[2])==np.matrix:
        ze = pose[2].item()
    else:
        ze = pose[2]
    if type(pose[3])==np.matrix:
        phi = ((pose[3].item() - 90) * D2R)
    else:
        phi = ((pose[3] - 90) * D2R)
    
    theta1 = np.math.atan2(ye, xe) - (np.pi / 2)

    l6 = 174.15
    xc = xe + l6 * np.math.cos(phi) * np.math.sin(theta1)
    yc = ye - l6 * np.math.cos(phi) * np.math.cos(theta1)
    zc = ze + l6 * np.math.sin(phi)

    r = np.math.sqrt(xc ** 2 + yc ** 2)
    l1 = 103.91
    s = zc - l1

    l2 = np.sqrt(200**2 + 50**2)
    l3 = 200

    temp = (r**2 + s**2 - l2**2 - l3**2)/(2*l2*l3)
    if (np.abs(temp)>1):
        return np.array([float('NaN'), float('NaN'), float('NaN'), float('NaN')])

    theta3 = - np.arccos(temp)
    theta2 = np.math.atan2(s, r) - np.math.atan2((l3 * np.math.sin(theta3)), l2 + l3 * np.math.cos(theta3))
    theta4 = - phi - (theta2 + theta3)

    theta1 = clamp(theta1)
    theta2 = clamp(np.pi / 2.0 - theta2 - l2_offset)
    theta3 = clamp(np.pi / 2.0 + theta3 - l2_offset)
    theta4 = clamp(theta4)
    
    return np.array([theta1, theta2, theta3, theta4])