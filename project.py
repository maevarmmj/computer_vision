import math

import cv2 as cv
import numpy as np

def calculate_moment(mask):
    moments = cv.moments(mask)

    return moments

cap = cv.VideoCapture('Video/Mousse_new.mp4')
frameTime = 50
trajectoire = []


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
    pts1 = np.float32([point_hg, point_hd, point_bg, point_bd]) # points réels

    point_hg_new = (0, 0)
    point_hd_new = (1080, 0)
    point_bg_new = (0, 621)
    point_bd_new = (1080, 621)
    pts2 = np.float32([point_hg_new, point_hd_new, point_bg_new, point_bd_new]) # points sur la vidéo (pour transformation)

    M = cv.getPerspectiveTransform(pts1, pts2)
    transform = cv.warpPerspective(frame, M, (1080, 621)) # facteur 1,85


    cv.imshow('transform', transform)


    # Convert BGR to HSV
    hsv = cv.cvtColor(transform, cv.COLOR_BGR2HSV)

    # define range of red color in HSV
    upper_red = np.array([178, 179, 166])
    lower_red = np.array([2, 215, 102])

    mask = cv.inRange(hsv, lower_red-80, upper_red+80)

    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #
    #
    #
    res = cv.bitwise_and(transform, transform, mask=mask)

    if contours: # contours found?

        largest_contour = max(contours, key=cv.contourArea)

        mask_largest_area = np.zeros_like(mask)

        cv.drawContours(mask_largest_area, [largest_contour], -1, 255, cv.FILLED)

        mask = mask_largest_area

        # # Bounding Box
        x, y, w, h = cv.boundingRect(largest_contour)
        cv.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)

        moment_mask = calculate_moment(mask)


        # centroid point
        cX = int(moment_mask["m10"] / moment_mask["m00"])
        cY = int(moment_mask["m01"] / moment_mask["m00"])
        centroid = (cX,cY)
        cv.circle(res, centroid, 4, (255, 0, 0), -1)
        #
        # mu20 = moment_mask['mu20'] / moment_mask['m00']
        # mu02 = moment_mask['mu02'] / moment_mask['m00']
        # mu11 = moment_mask['mu11'] / moment_mask['m00']
        #
        # angle = math.degrees(0.5 * math.atan2(2 * mu11, mu20 - mu02))
        #
        # common_part = (mu20 + mu02)
        # diff_part = math.sqrt((4 * mu11 ** 2) + ((mu20 - mu02) ** 2))
        # a = math.sqrt(common_part + diff_part)
        # b = math.sqrt(common_part - diff_part)
        #
        # l = int(math.sqrt(8 * a)) * 4
        # w = int(math.sqrt(8 * b)) * 4
        #
        # axes = (l,w)
        # color = (255, 0, 0)
        #
        # # Ellipse moyen
        # cv.circle(res, centroid, radius=5, color=color, thickness=-1)
        # cv.ellipse(res, centroid, axes, angle, 0, 360, color, 2)

        trajectoire.append(centroid)
        for point in trajectoire:
            cv.circle(res, point, 3, (0, 255, 255), -1)  # Trajectoire avec pts jaunes



    else:
        mask = np.zeros_like(mask)


    cv.imshow('frame', frame)
    cv.imshow('res', res)

    k = cv.waitKey(5) & 0xFF
    if cv.waitKey(frameTime) & 0xFF == ord('q'):
        break


cap.release()
cv.destroyAllWindows()