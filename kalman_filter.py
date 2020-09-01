
import numpy as np

class KalmanFilter:

    def __init__(self):

        self.dt = 0.05  # delta time
        self.A = np.array([[1, 0], [0, 1]])  # matrix in observation equations
        self.u = np.zeros((2, 1))  # previous state vector

        # (x,y) tracking object center
        self.b = np.array([[200], [200]])  # vector of observations

        self.P = np.diag((3, 2))  # covariance matrix
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])  # state transition mat

        self.Q = np.eye(self.u.shape[0])  # process noise matrix
        self.R = np.eye(self.b.shape[0])  # observation noise matrix
        self.lastResult = np.array([[200], [200]])

    def predict(self):

        # Predicted state estimate
        self.u = np.round(np.dot(self.F, self.u))
        # Predicted estimate covariance
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.lastResult = self.u  # same last predicted result

        return self.u

    def correct(self, b, flag):


        if not flag:  # update using prediction
            self.b = self.lastResult
        else:  # update using detection
            self.b = b

        C = np.dot(self.A, np.dot(self.P, self.A.T)) + self.R
        K = np.dot(self.P, np.dot(self.A.T, np.linalg.inv(C)))

        self.u = np.round(self.u + np.dot(K, (self.b - np.dot(self.A,self.u))))

        self.P = self.P - np.dot(K, np.dot(self.A, self.P))

        self.lastResult = self.u

        return self.u