import numpy as np
import cv2
import matplotlib.pyplot as plt
# This sets the program to ignore a divide error which does not affect the outcome of the program
#np.seterr(divide='ignore', invalid='ignore')

def main():

    img1 = cv2.imread('cubecut1.tif',cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('cubecut2.tif',cv2.IMREAD_GRAYSCALE)

    u,v =  lucas_kanade(img1,img2,7)
    quiver(u,v,img1)

    flow_map = compute_flow_map(u,v,img1,1)
    cv2.imshow('flowmap',flow_map)
    cv2.waitKey(0)
    # cv2.imshow('leveldown',reduce(img1,2))
    # cv2.waitKey(0)
    # cv2.imshow('levelup',expand(img1,2))
    # cv2.waitKey(0)
    return


def reduce(image, level=1):
    result = np.copy(image)

    for _ in range(level - 1):
        result = cv2.pyrDown(result)

    return result

def expand(image, level=1):
    return cv2.pyrUp(np.copy(image))

def compute_flow_map(u, v,img1,gran):
    flow_map = np.array(img1,dtype = np.uint8)

    for y in range(flow_map.shape[0]):
        for x in range(flow_map.shape[1]):

            if y % gran == 0 and x % gran == 0:
                dx = 10 * int(u[y, x])
                dy = 10 * int(v[y, x])

                if dx > 0 or dy > 0:
                    cv2.arrowedLine(flow_map, (x, y), (x + dx, y + dy), 0, 1)

    return flow_map


'''
TODO: Add comments for this method
'''
def lucas_kanade(im1, im2, win=7):

    im1 = np.array(im1, dtype = float)
    im2 = np.array(im2, dtype = float)
    Ix = np.zeros(im1.shape)
    Iy = np.zeros(im1.shape)
    It = np.zeros(im1.shape)

    Ix[1:-1, 1:-1] = (im1[1:-1, 2:] - im1[1:-1, :-2]) / 2
    Iy[1:-1, 1:-1] = (im1[2:, 1:-1] - im1[:-2, 1:-1]) / 2
    It[1:-1, 1:-1] = im1[1:-1, 1:-1] - im2[1:-1, 1:-1]

    params = np.zeros(im1.shape + (5,))
    params[..., 0] = cv2.GaussianBlur(Ix * Ix, (5, 5), 3)
    params[..., 1] = cv2.GaussianBlur(Iy * Iy, (5, 5), 3)
    params[..., 2] = cv2.GaussianBlur(Ix * Iy, (5, 5), 3)
    params[..., 3] = cv2.GaussianBlur(Ix * It, (5, 5), 3)
    params[..., 4] = cv2.GaussianBlur(Iy * It, (5, 5), 3)

    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -cum_params[2 * win + 1:, :-1 - 2 * win] - cum_params[:-1 - 2 * win, 2 * win + 1:] + cum_params[:-1 - 2 * win, :-1 - 2 * win])

    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)

    Ixx = win_params[..., 0]
    Iyy = win_params[..., 1]
    Ixy = win_params[..., 2]
    Ixt = -win_params[..., 3]
    Iyt = -win_params[..., 4]

    M_det = Ixx * Iyy - Ixy ** 2
    temp_u = Iyy * (-Ixt) + (-Ixy) * (-Iyt)
    temp_v = (-Ixy) * (-Ixt) + Ixx * (-Iyt)
    op_flow_x = np.where(M_det != 0, temp_u / M_det, 0)
    op_flow_y = np.where(M_det != 0, temp_v / M_det, 0)

    u[win + 1: -1 - win, win + 1: -1 - win] = op_flow_x[:-1, :-1]
    v[win + 1: -1 - win, win + 1: -1 - win] = op_flow_y[:-1, :-1]

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



main()