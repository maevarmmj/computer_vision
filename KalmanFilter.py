import numpy as np

class KalmanFilter(object):
    def __init__(self, dt, point, pixel_to_meters): # Ajout de pixel_to_meters en argument
        self.dt=dt
        self.pixel_to_meters = pixel_to_meters

        # ----------- Vecteur d'etat initial (x, y, vx, vy, ax, ay) ----------------
        self.E=np.matrix([[point[0]], [point[1]], [0], [0], [0], [0]])

        # ----------- Matrice de transition ------------------
        self.A=np.matrix([[1, 0, self.dt, 0, 0.5*self.dt**2, 0],
                          [0, 1, 0, self.dt, 0, 0.5*self.dt**2],
                          [0, 0, 1, 0, self.dt, 0],
                          [0, 0, 0, 1, 0, self.dt],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]])

        self.H=np.matrix([[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0]])

        self.coeffQ = 1e-15
        self.Q=np.matrix([[self.coeffQ, 0, 0, 0, 0, 0],
                          [0, self.coeffQ, 0, 0, 0, 0],
                          [0, 0, self.coeffQ, 0, 0, 0],
                          [0, 0, 0, self.coeffQ, 0, 0],
                          [0, 0, 0, 0, self.coeffQ, 0],
                          [0, 0, 0, 0, 0, self.coeffQ]])

        self.coeffR = 1e-15
        self.R=np.matrix([[self.coeffR, 0],
                          [0, self.coeffR]])

        self.P=np.eye(self.A.shape[1])

    def predict(self):
        self.E=np.dot(self.A, self.E)
        # ---------- Calcul de la covariance de l'erreur ----------
        self.P=np.dot(np.dot(self.A, self.P), self.A.T)+self.Q

        # ---------- Ajout de la gravité ----------
        gravity_pixel_per_frame_sq = 9.81 / self.pixel_to_meters * (self.dt**2)
        self.E[5, 0] = self.E[5, 0] + gravity_pixel_per_frame_sq # On ajoute la gravité à ay

        return self.E

    def update(self, z):
        # -------- Calcul du gain de Kalman --------------
        S=np.dot(self.H, np.dot(self.P, self.H.T))+self.R
        K=np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # -------- Correction / innovation ---------------
        self.E=np.round(self.E+np.dot(K, (z-np.dot(self.H, self.E))))
        I=np.eye(self.H.shape[1])
        self.P=(I-(K*self.H))*self.P

        return self.E
