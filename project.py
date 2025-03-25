import math
import time
import cv2 as cv
import numpy as np

def calculate_moment(mask):
    moments = cv.moments(mask)

    return moments

cap = cv.VideoCapture('Video/Mouse_new.mp4')
frameTime = 1
trajectoire = []
PIXEL_TO_METERS = 0.00185 # coeff pour la vitesse plus tard
previous_centroid = None  # Dernière position connue de la balle
previous_time = None  # Dernier timestamp
g = 9.81  # gravité




while cap.isOpened():
    ret, frame = cap.read()
    frame = cv.resize(frame, (0,0), fx=0.7, fy=0.7)


    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # POINTS POUR HOMOGRAPHIE
    point_hg = (120, 5)
    point_hd = (690, 330)
    point_bg = (20, 525)
    point_bd = (675, 590)
    pts1 = np.float32([point_hg, point_hd, point_bg, point_bd]) # points du tableau sur la video

    point_hg_new = (0, 0)
    point_hd_new = (1080, 0)
    point_bg_new = (0, 621)
    point_bd_new = (1080, 621)
    pts2 = np.float32([point_hg_new, point_hd_new, point_bg_new, point_bd_new]) # points finaux

    M = cv.getPerspectiveTransform(pts1, pts2)
    transform = cv.warpPerspective(frame, M, (1080, 621)) # facteur 1,85


    # Convert BGR to HSV
    hsv = cv.cvtColor(transform, cv.COLOR_BGR2HSV)

    # define range of red color in HSV
    upper_red = np.array([178, 179, 166])
    lower_red = np.array([2, 215, 102])

    mask = cv.inRange(hsv, lower_red-80, upper_red+80)

    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    res = cv.bitwise_and(transform, transform, mask=mask)

    if contours: # contours found?

        largest_contour = max(contours, key=cv.contourArea)

        mask_largest_area = np.zeros_like(mask)

        cv.drawContours(mask_largest_area, [largest_contour], -1, 255, cv.FILLED)

        mask = mask_largest_area

        # Bounding Box
        x, y, w, h = cv.boundingRect(largest_contour)
        cv.rectangle(transform, (x, y), (x + w, y + h), (0, 255, 0), 2)

        moment_mask = calculate_moment(mask)


        # centroid point
        cX = int(moment_mask["m10"] / moment_mask["m00"])
        cY = int(moment_mask["m01"] / moment_mask["m00"])
        centroid = (cX,cY)
        cv.circle(transform, centroid, 4, (255, 0, 0), -1)

        mu20 = moment_mask['mu20'] / moment_mask['m00']
        mu02 = moment_mask['mu02'] / moment_mask['m00']
        mu11 = moment_mask['mu11'] / moment_mask['m00']

        angle = math.degrees(0.5 * math.atan2(2 * mu11, mu20 - mu02))

        trajectoire.append(centroid)
        for point in trajectoire:
            cv.circle(transform, point, 3, (0, 255, 255), -1)  # Trajectoire avec pts jaunes

        if len(trajectoire) > 1:
            (x1, y1) = trajectoire[0]
            (x2, y2) = trajectoire[1]

            angle_init = math.atan2(y2 - y1, x2 - x1)

            delta_t = frameTime
            v0_pix = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            v0_meters = v0_pix * PIXEL_TO_METERS
            v0 = v0_meters / delta_t

            print(f"Vitesse initiale : {v0:.2f} m/s, Angle initial : {math.degrees(angle_init):.2f}°")


        # VITESSE
        current_time = time.time()  # Temps actuel en secondes
        if previous_centroid is not None and previous_time is not None:
            x_prev, y_prev = previous_centroid
            distance_pixels = math.sqrt((cX - x_prev) ** 2 + (cY - y_prev) ** 2)
            distance_meters = distance_pixels * PIXEL_TO_METERS

            delta_t = current_time - previous_time  # Temps écoulé en secondes

            if delta_t > 0:
                speed_mps = distance_meters / delta_t  # m/s
                print(f"Vitesse = {speed_mps:.2f} m/s")

                cv.putText(transform, f"{speed_mps:.2f} m/s", (cX + 20, cY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv.putText(transform, f"{angle:.2f} deg", (cX + 20, cY - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


        previous_centroid = centroid
        previous_time = current_time

    else:
        mask = np.zeros_like(mask)

    # cv.imshow('frame', frame)
    cv.imshow('res', transform)

    k = cv.waitKey(5) & 0xFF
    if cv.waitKey(frameTime) & 0xFF == ord('q'):
        break


cap.release()
cv.destroyAllWindows()