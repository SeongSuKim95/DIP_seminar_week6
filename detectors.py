
import numpy as np
import cv2

debug = 1

class Detectors:

    def Detect(self, frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if (debug == 1):
                cv2.imshow('gray', gray)

            ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            # if (debug == 0):
            #     cv2.imshow('thresh', thresh)

            centers = []
            # vector of object centroids in a frame
            # we only care about centroids with size of bug in this example
            # recommended to be tunned based on expected object size for
            # improved performance
            blob_radius_thresh = 1
            # Find centroid for each valid contours
            for cnt in contours:
                try:
                    # Calculate and draw circle
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    centeroid = (int(x), int(y))
                    radius = int(radius)+3
                    if (radius > blob_radius_thresh):
                        cv2.circle(frame, centeroid, radius, (255, 255, 255), 1)
                        b = np.array([[x], [y]])
                        centers.append(np.round(b))
                except ZeroDivisionError:
                    pass

            return centers