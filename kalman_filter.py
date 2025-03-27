import numpy as np

class KalmanFilter:
    def __init__(self, dt):
        self.F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
        self.Q = np.eye(4) * 100
        self.R = np.eye(2) * 500
        self.x = np.array([[0.0], [0.0], [0.0], [0.0]])
        self.P = np.eye(4) * 1000.0

    # ------------ Prédiction de l'état et de sa covariance -----------
    def predict(self, u=None): # u est maintenant optionnel
        if u is None:
            self.x = np.dot(self.F, self.x)
        else:
            self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    # -------- Compute le gain de Kalman + MAJ de l'état estimé et de la matrice de covariance ------------------
    def update(self, z):
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        return self.x



