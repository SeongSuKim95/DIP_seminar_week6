import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
def main( ):
    #
    # cap = cv2.VideoCapture('project2.avi')
    # ret, frame = cap.read()
    img2 = cv2.imread('28.bmp', cv2.IMREAD_GRAYSCALE)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    fgmask = fgbg.apply(img2)
    # edges = cv2.Canny(fgmask, 50, 190, 3)



    ret, thresh = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    centers = []  # vector of object centroids in a frame
    # we only care about centroids with size of bug in this example
    # recommended to be tunned based on expected object size for
    # improved performance
    blob_radius_thresh = 0
    # Find centroid for each valid contours
    for cnt in contours:
        try:
            # Calculate and draw circle
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            print(radius)
            centeroid = (int(x), int(y))
            radius = int(radius)
            if (radius > blob_radius_thresh):
                cv2.circle(img2, centeroid, radius, (0, 255, 0), 2)
                b = np.array([[x], [y]])
                centers.append(np.round(b))
        except ZeroDivisionError:
            pass

    # print(contours)
    # image = cv2.drawContours(img2,contours,0,2)
    # plt.subplot(1, 2, 2)
    # plt.imshow(image, cmap='bone')
    # plt.title("dd")
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    #cap.release()
    return

def test():

    class Track:

        def __init__(self, prediction, trackIdCount):
            self.track_id = trackIdCount  # identification of each track object
              # KF instance to track this object
            self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
            self.skipped_frames = 0  # number of frames skipped undetected
            self.trace = []  # trace path

    class Tracker:

        def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length, trackIdCount):

            self.dist_thresh = dist_thresh
            self.max_frames_to_skip = max_frames_to_skip
            self.max_trace_length = max_trace_length
            self.trackIdCount = trackIdCount
            self.tracks = []

        def Update(self, detections):
            # Create tracks if no tracks vector found
            if (len(self.tracks) == 0):
                for i in range(len(detections)):
                    track = Track(detections[i], self.trackIdCount)
                    self.trackIdCount += 1
                    self.tracks.append(track)

            # Calculate cost using sum of square distance between predicted vs detected centroids
            N = len(self.tracks)
            M = len(detections)
            print(N,M)
            print(self.tracks, detections)

            cost = np.zeros(shape=(N, M))  # Cost matrix

            for i in range(len(self.tracks)):
                for j in range(len(detections)):
                    try:
                        diff = self.tracks[i].prediction - detections[j]
                        distance = np.sqrt(diff[0][0] * diff[0][0] + diff[1][0] * diff[1][0])  # diff[0][0] -> x of diff, diff[1][0] -> y of diff
                        cost[i][j] = distance
                    except:
                        pass
            # Let's average the squared ERROR
            cost = (0.5) * cost
            # Using Hungarian Algorithm assign the correct detected measurements to predicted tracks
            print(cost)

    tracker = Tracker(200, 10, 5, 100)
    tracker.Update([[[21],[19]],[[20],[11]]])
    return
def printshape():

    b = np.array([[2],[2]])
    a = np.eye(b.shape[0])
    print(a)

    return
printshape()