import math
import cv2 as cv
import numpy as np
from KalmanFilter import KalmanFilter


# --------- Calcul du moment pour la vitesse instantanée -------
def calculate_moment(mask):
    moments = cv.moments(mask)
    return moments


def etude_video(video_path, upper_range, lower_range):
    # --------- Initialisation des paramètres -----------------
    cap = cv.VideoCapture(video_path)
    frameRate = 30.0  # Frame / s
    frameTime = 1.0 / frameRate
    angle_init = 0
    v0 = 0

    v0y = 0
    v0x = 0
    x1 = 0
    y1 = 0
    trajectoire = []

    PIXEL_TO_METERS = 0.00185 # Coeff de proportionnalité entre les dimensions réelles et les dimensions utilisées dans la vidéo
    frame_counter = 0
    previous_centroid = None  # Dernière position connue de la balle
    previous_frame_time = None  # Dernier timestamp
    g = 9.81  # Coefficient gravité
    compteur = 0

    # -------- Initialisation du filtre de Kalman d'ordre 2 --------
    initial_position = (0, 0)
    kalman_filter = KalmanFilter(frameTime, initial_position, PIXEL_TO_METERS)
    predicted_position = initial_position # Position prédite initialisée

    # ------------ Tant que la vidéo est lue ------------------
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv.resize(frame, (0,0), fx=0.7, fy=0.7)

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # ------------- Partie sur l'homographie -----------------
        point_hg = (120, 5)
        point_hd = (690, 330)
        point_bg = (20, 525)
        point_bd = (675, 590)
        pts1 = np.float32([point_hg, point_hd, point_bg, point_bd]) # Points du tableau sur la video

        point_hg_new = (0, 0)
        point_hd_new = (1080, 0)
        point_bg_new = (0, 621)
        point_bd_new = (1080, 621)
        pts2 = np.float32([point_hg_new, point_hd_new, point_bg_new, point_bd_new]) # Points finaux

        M = cv.getPerspectiveTransform(pts1, pts2)
        transform = cv.warpPerspective(frame, M, (1080, 621)) # Facteur 1,85 entre les dim réelles et finales

        # ---------- Partie détection de l'objet -------------
        hsv = cv.cvtColor(transform, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_range, upper_range)
        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # -------- Prédiction de la position à chaque frame (avant même la détection) --------
        predicted_state = kalman_filter.predict()
        predicted_position = (int(predicted_state[0, 0]), int(predicted_state[1, 0])) # Extraction de x et y prédits


        # ----------- Si les contours de l'objet sont bien détectés -------------
        if contours:

            largest_contour = max(contours, key=cv.contourArea)
            mask_largest_area = np.zeros_like(mask)
            cv.drawContours(mask_largest_area, [largest_contour], -1, 255, cv.FILLED)
            mask = mask_largest_area

            # ---------- Création de la boîte englobante -----------
            x, y, w, h = cv.boundingRect(largest_contour)
            cv.rectangle(transform, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # ----------- Calcul du centroïde + trajectoire ---------------------
            moment_mask = calculate_moment(mask)
            cX = int(moment_mask["m10"] / moment_mask["m00"])
            cY = int(moment_mask["m01"] / moment_mask["m00"])
            centroid = (cX,cY)
            cv.circle(transform, centroid, 4, (255, 0, 0), -1) # Cercle bleu pour la détection

            # -------- Mise à jour du filtre de Kalman avec la détection --------
            measurement = np.matrix([[cX], [cY]]) # Mesure = centroïde détecté
            kalman_filter.update(measurement)

            # -------- Mise à jour de la position prédite (après l'update, le filtre a une meilleure estimation) -----
            predicted_state = kalman_filter.predict() # On refait une prédiction après l'update pour avoir la position la plus à jour
            predicted_position = (int(predicted_state[0, 0]), int(predicted_state[1, 0]))

            trajectoire.append(centroid) # On ajoute le centroïde détecté à la trajectoire (pour la visualisation)
            for point in trajectoire:
                cv.circle(transform, point, 3, (0, 255, 255), -1)  # Trajectoire en jaune


            # ----------- Calcul de l'angle initial de la balle -----------------
            # v0 : vitesse initiale quand la balle ARRIVE sur la vidéo (pas la vraie vitesse initiale)
            if len(trajectoire) > 1:
                (x1, y1) = trajectoire[0]
                (x2, y2) = trajectoire[1]

                angle_init = - math.atan2((y2 - y1), (x2 - x1))

                for i in range (len(trajectoire)-1):
                    if trajectoire[i-1] > trajectoire[i]:
                        (x2, y2) = trajectoire[i-1]
                        hauteur_max = y2 * PIXEL_TO_METERS
                        v0 = math.sqrt(2*g*hauteur_max)/math.sin(angle_init)

                if compteur == 0:
                    print(f"Vitesse initiale : {v0:.2f} m/s, Angle initial : {math.degrees(angle_init):.2f}°")
                    compteur += 1

            # ------------ Calcul de l'angle et de la vitesse instantanée ------------------

            mu20 = moment_mask['mu20'] / moment_mask['m00']
            mu02 = moment_mask['mu02'] / moment_mask['m00']
            mu11 = moment_mask['mu11'] / moment_mask['m00']

            angle = math.degrees(0.5 * math.atan2(2 * mu11, mu20 - mu02))

            current_frame_time = frame_counter * frameTime
            if previous_centroid is not None and previous_frame_time is not None:
                x_prev, y_prev = previous_centroid
                delta_t = current_frame_time - previous_frame_time  # Temps en secondes
                if delta_t > 0:
                    v_x = (x_prev - cX) / delta_t
                    v_y = (y_prev - cY) / delta_t
                    v_x = v_x * PIXEL_TO_METERS
                    v_y = v_y * PIXEL_TO_METERS
                    v = math.sqrt(v_x ** 2 + v_y ** 2)
                    print(f"Vitesse instantanée : {v:.2f} m/s")
                    cv.putText(transform, f"{v:.2f} m/s", (cX + 20, cY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv.putText(transform, f"{angle:.2f} deg", (cX + 20, cY - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            frame_counter += 1
            previous_centroid = centroid
            previous_frame_time = current_frame_time

        # -------- Affichage de la position prédite (même si pas de détection) --------
        cv.circle(transform, predicted_position, 5, (0, 0, 255), -1) # Cercle rouge pour la position prédite


        cv.imshow('resultat', transform)
        k = cv.waitKey(5) & 0xFF
        if cv.waitKey(int(frameTime * 1000)) & 0xFF == ord('q'):
            break


    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    # BALLE ROUGE
    balle_rouge = "Video/Mousse_new.mp4"
    upper_red = np.array([255, 255, 246])
    lower_red = np.array([0, 135, 22])

    # BALLE DE RUGBY
    balle_rugby = "Video/rugby_tab.mp4"
    lower_rugby = np.array([90, 200, 30])
    upper_rugby = np.array([110, 255, 120])

    # BALLE JAUNE
    balle_jaune = "Video/tennis_tab.mp4"
    lower_yellow = np.array([20, 80, 100])
    upper_yellow = np.array([40, 255, 255])

    etude_video(balle_rouge, upper_red, lower_red)

