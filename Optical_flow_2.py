import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt
def main():

    img1 = cv2.imread('cubecut1.tif',cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('cubecut2.tif',cv2.IMREAD_GRAYSCALE)
    u,v = optical_flow(img1,img2,4,1e-2)
    flow_map = compute_flow_map(u,v,img1,1)
    #img_re = Reconstruction(img1,u,v)
    cv2.imshow('REcon',flow_map)
    cv2.waitKey(0)

    return

def optical_flow(I1g, I2g, window_size, tau=1e-2):

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
    w = int(window_size/2) # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255. # normalize pixels
    I2g = I2g / 255. # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size

    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            b = np.reshape(It, (It.shape[0], 1))  # get b here
            A = np.vstack((Ix, Iy)).T  # get A here

            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b)  # get velocity here
                u[i, j] = nu[0]
                v[i, j] = nu[1]
    u = np.array(np.round(u),dtype = 'int16')
    v = np.array(np.round(v),dtype = 'int16')
    return u,v

def quiver(u, v, img1):
    flow_map = np.array(img1, dtype = np.uint8)
    y,x = flow_map.shape
    X= np.arange(x)
    Y = y-1- np.arange(y)


    plt.figure()
    plt.quiver(X,Y,u,v,angles='xy')
    plt.show()

    # max_u = np.max(u)
    # # # min_u = np.min(u)
    # max_v = np.max(v)
    # # # min_v = np.min(u)
    # u = (u/max_u) * 10
    # v = (v/max_v) * 10

    # for y in range(flow_map.shape[0]):
    #     for x in range(flow_map.shape[1]):
    #             dx = int(u[y, x])
    #             dy = int(v[y, x])
    #             if dx > 0 or dy > 0:
    #                 cv2.arrowedLine(flow_map, (x, y), (x+dx , y+dy), 0, 1)

    return

def compute_flow_map(u, v,img1,gran):
    flow_map = np.array(img1,dtype = np.uint8)

    for y in range(flow_map.shape[0]):
        for x in range(flow_map.shape[1]):

            if y % gran == 0 and x % gran == 0:
                dx =u[y, x]
                dy =v[y, x]

                if dx > 0 or dy > 0:
                    cv2.arrowedLine(flow_map, (x, y), (x + dx, y + dy), 0, 1)

    return flow_map


def Reconstruction(imgt_1, u, v):
    imgt_1 = np.array(imgt_1,dtype= 'int16')
    y,x = imgt_1.shape

    for i in range(y):
        for j in range(x):
          if u[i,j] != 0 or v[i,j] != 0 :

            I = i + u[i,j]
            J = j + v[i,j]
            if I <= y-1 and j <= x-1:
                imgt_1[i][j] = imgt_1[I][J]
    imgt_1 = np.array(imgt_1, dtype=np.uint8)

    return imgt_1
main()
