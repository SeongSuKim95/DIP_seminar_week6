
import cv2

from detectors import Detectors
from tracker import Tracker


def main():

    # Create opencv video capture object
    cap = cv2.VideoCapture('project2.avi')

    # Create Object Detector
    detector = Detectors()
    # Create Object Tracker
    tracker = Tracker(40, 8, 5, 100)

    # Variables initialization

    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),(0, 255, 255), (255, 0, 255), (255, 127, 255),(127, 0, 255), (127, 0, 127)]

    out = cv2.VideoWriter('Tracking2_wait8.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, (200,200))


    while(True):
        ret, frame = cap.read()
        if ret== True :
            centers = detector.Detect(frame)

            # If centroids are detected then track them
            if (len(centers) >= 0):
                # Track object using Kalman Filter
                tracker.Update(centers)
                # For identified object tracks draw tracking line
                # Use various colors to indicate different track_id
                for i in range(len(tracker.tracks)):
                    if (len(tracker.tracks[i].trace) > 1):
                        for j in range(len(tracker.tracks[i].trace)-1):
                            # Draw trace line
                            x1 = tracker.tracks[i].trace[j][0][0]
                            y1 = tracker.tracks[i].trace[j][1][0]

                            x2 = tracker.tracks[i].trace[j+1][0][0]
                            y2 = tracker.tracks[i].trace[j+1][1][0]

                            clr = tracker.tracks[i].track_id % 9
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),track_colors[clr], 2)

                # Display the resulting tracking frame
                cv2.imshow('Tracking', frame)
                cv2.waitKey(100)
                out.write(frame)

        else:
            break;


    cap.release()
    cv2.destroyAllWindows()

    out.release()
if __name__ == "__main__":
    # execute main
    main()