import numpy as np
import cv2 as cv
import glob


def etalonnage():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7*10, 3), np.float32)
    objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane.

    images = glob.glob("Checkboard/*.jpg")

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, (10, 7), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (10, 7), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)

    cv.destroyAllWindows()

    # Calibration de la caméra
    if len(objpoints) > 0 and len(imgpoints) > 0:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Affichage des résultats
        print("\nMatrice intrinsèque de la caméra :\n", mtx)
        print("\nCoefficients de distorsion :\n", dist)

        # Sauvegarde des paramètres
        np.savez("calibration_data.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

        return mtx, dist
    else:
        print("\nErreur : Aucun damier détecté, calibration impossible.")
        return None, None

if __name__ == "__main__":
    etalonnage()