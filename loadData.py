import yaml
import numpy as np

def recover_homogeneous_transform_svd(m, d):
    ''' 
    finds the rigid body transform that maps m to d: 
    d == np.dot(m,R) + T
    http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
    '''
    # calculate the centroid for each set of points
    d_bar = np.sum(d, axis=0) / np.shape(d)[0]
    m_bar = np.sum(m, axis=0) / np.shape(m)[0]

    # we are using row vectors, so tanspose the first one
    # H should be 3x3, if it is not, we've done this wrong
    H = np.dot(np.transpose(d - d_bar), m - m_bar)
    [U, S, V] = np.linalg.svd(H)

    R = np.matmul(V, np.transpose(U))
    # if det(R) is -1, we've made a reflection, not a rotation
    # fix it by negating the 3rd column of V
    if np.linalg.det(R) < 0:
        V = [1, 1, -1] * V
        R = np.matmul(V, np.transpose(U))
    T = d_bar - np.dot(m_bar, R)
    return np.transpose(np.column_stack((np.row_stack((R, T)), (0, 0, 0, 1))))

#with open('camera_calibration/calibrationdata/ost.yaml', 'r') as file:
#    data = yaml.safe_load(file)

with open('camera_calibration/factory.yaml', 'r') as file:
    data = yaml.safe_load(file)

#dataList = data['camera_matrix']['data']

dataList = data['K']

K = np.zeros((3,3))

for i in range(len(dataList)):
    K[i/3][i%3] = dataList[i]

'''for i in range(5):

pixels = np.array([],)

# pixels to camera in mm

P2C = d *  np.linalg.inv(K)

P2C_H = np.append(P2C, [[0,0,1]], axis = 0)

# camera to World

A_svd = recover_homogeneous_transform_svd(points_world, points_camera)

#print(P2W)'''

pixelMatrix = np.array([[671, 388, 976], [429, 293, 945], [403, 605, 866], [932, 283, 902], [958, 614, 823]])

worldMatrix = np.array([[0,175,0],[-250, 275, 38.58],[-250, -25, 114.17],[250, 275, 76.14],[250, -25, 152.72]]).T
            
cameraMatrix = np.zeros_like(worldMatrix)


# pixels to camera

for i in range(pixelMatrix.shape[0]):
    P2C = pixelMatrix[i, 2] *  np.linalg.inv(K)

    pixel = np.array([[pixelMatrix[i, 0]],[pixelMatrix[i, 1]],[1]])

    camera = np.matmul(P2C, pixel)

    #camera = np.append(camera, [[1]], axis = 0)

    cameraMatrix[:, i] = camera.reshape(1, -1)


# camera to World

H = recover_homogeneous_transform_svd(worldMatrix.T, cameraMatrix.T)

H_inv = np.linalg.inv(H)

depth = 1
z = depth
x = 2
y = 3

P2C = z *  np.linalg.inv(K)

pixel = np.array([[x],[y],[1]])

camera = np.matmul(P2C, pixel)

camera = np.append(camera, [[1]], axis = 0)

world = np.matmul(H_inv, camera)
