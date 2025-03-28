import math
import cv2 as cv
import numpy as np
from KalmanFilter import KalmanFilter

# --------- Tracé de parabole -------------------

def trace_parabole(trajectoire):
    pts_parabole = None
    coeffs = None

    if len(trajectoire) > 2:
        listcX, listcY = [],[]
        for centroid in trajectoire:
            listcX.append(centroid[0])
            listcY.append(centroid[1])
        if (len(listcX) > 2):
            coeffs = np.polyfit(listcX, listcY, 2)
        if coeffs is not None:
            poly = np.poly1d(coeffs)
            x_range = np.linspace(0, 1080, 100)
            y_range = poly(x_range)
            pts_parabole = np.array([x_range, y_range], dtype=np.int32).T

    return pts_parabole


# --------- Calcul du moment pour la vitesse instantanée -------
def calculate_moment(mask):
    moments = cv.moments(mask)
    return moments


def etude_video(video_path, upper_range, lower_range):
    # --------- Initialisation des paramètres -----------------
    cap = cv.VideoCapture(video_path)
    frameRate = 30.0  # Frame / s
    frameTime = 1.0 / frameRate
    v0 = 0
    xmax, ymax = 0, 0
    area_min = 0
    angle_init = 0
    distance_totale = 0
    distance_predite = distance_totale
    hauteur = 1 # Hauteur entre le sol et le bas du tableau en m
    trajectoire = []
    trajectoire_predite = []

    PIXEL_TO_METERS = 0.00185 # Coeff de proportionnalité entre les dimensions réelles et les dimensions utilisées dans la vidéo
    frame_counter = 0
    previous_centroid = None  # Dernière position connue de la balle
    previous_frame_time = None  # Dernier timestamp
    g = 9.81  # Coefficient gravité
    compteur = 0
    compteur_v0 = 0

    # -------- Initialisation du filtre de Kalman d'ordre 2 --------
    initial_position = (0, 0)
    kalman_filter = KalmanFilter(frameTime, initial_position, PIXEL_TO_METERS)
    predicted_position = initial_position # Position prédite initialisée

    prediction_points_count = 30  # Prédiction de 30 points après la fin de détection de la balle
    future_trajectory_points = []  # Liste pour stocker les points de la trajectoire future

    # ------------ Tant que la vidéo est lue ------------------
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv.resize(frame, (0,0), fx=0.7, fy=0.7)

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # ------------- Partie sur l'homographie -----------------
        point_hg = (125, 10)
        point_hd = (685, 332)
        point_bg = (25, 522)
        point_bd = (672, 590)
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
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

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
            area = cv.contourArea(largest_contour)

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
                cv.circle(transform, point, 3, (52, 235, 208), -1)  # Trajectoire en jaune


            # ----------- Calcul de l'angle initial de la balle -----------------
            if len(trajectoire) > 1:
                (x1, y1) = trajectoire[0]
                (x2, y2) = trajectoire[1]

                angle_init = - math.atan2((y2 - y1), (x2 - x1))
                x_prev, y_prev = previous_centroid
                x_current, y_current = centroid
                if y_prev < y_current and compteur == 0:
                    (xmax, ymax) = previous_centroid
                    hauteur_max = ymax * PIXEL_TO_METERS
                    compteur += 1
                if (xmax, ymax) != (0,0):
                    cv.line(transform, (xmax, ymax), (xmax, 621), (255, 0, 255), 2)

                # ---------- Tracé de la trajecetoire -----------

            parabole = trace_parabole(trajectoire)
            cv.polylines(transform, [parabole], False, (52, 235, 208), 1)

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
                    if compteur_v0 == 0 :
                        v0 = v
                        print(f"Vitesse initiale : {v:.2f} m/s")
                        compteur_v0 += 1
                    else:
                        print(f"Vitesse instantanée : {v:.2f} m/s")
                    cv.putText(transform, f"{v:.2f} m/s", (cX + 20, cY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv.putText(transform, f"{angle:.2f} deg", (cX + 20, cY - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            frame_counter += 1
            previous_centroid = centroid
            previous_frame_time = current_frame_time

            future_trajectory_points = []

            # -------- Affichage de la position prédite (même si pas de détection) --------

        else:
            future_trajectory_points = []
            temp_kalman_filter = KalmanFilter(kalman_filter.dt, (kalman_filter.E[0, 0], kalman_filter.E[1, 0]),
                                              kalman_filter.pixel_to_meters)  # Créer une copie temporaire du filtre
            temp_kalman_filter.E = kalman_filter.E.copy()  # Copier l'état actuel
            temp_kalman_filter.P = kalman_filter.P.copy()  # Copier la covariance actuelle

            for _ in range(prediction_points_count):
                predicted_future_state = temp_kalman_filter.predict()
                predicted_future_position = (int(predicted_future_state[0, 0]), int(predicted_future_state[1, 0]))
                future_trajectory_points.append(predicted_future_position)

            parabole_suite = trace_parabole(trajectoire) # Juste pour afficher la parabole même lorsque la balle n'est plus détectée
            cv.polylines(transform, [parabole_suite], False, (52, 235, 208), 1)
            if (xmax, ymax) != (0, 0):
                cv.line(transform, (xmax, ymax), (xmax, 621), (255, 0, 255), 2)

        # -------- Affichage de la position prédite (temps réel) --------
        cv.circle(transform, predicted_position, 5, (0, 0, 255), -1) # En rouge

        # -------- Affichage de la trajectoire future prédite (points plus espacés) --------
        for future_point in future_trajectory_points:
            cv.circle(transform, future_point, 3, (0, 255, 0), -1) # En vert


        # ---------- Calcul distance parcourue par la balle + distance prédite par Kalman ------------
        distance_totale = v0 / g * math.cos(angle_init) * (
                v0 * math.sin(angle_init) + math.sqrt((v0 * math.sin(angle_init)) ** 2 + 2 * g * hauteur))
        if distance_totale != 0:
            print(f"Distance totale parcourue par la balle : {distance_totale:.2f} m")

        cv.imshow('resultat', transform)
        cv.waitKey(200) & 0xFF
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
    upper_rugby = np.array([110, 255, 120])
    lower_rugby = np.array([90, 200, 30])

    # BALLE JAUNE
    balle_jaune = "Video/tennis_tab.mp4"
    upper_yellow = np.array([40, 255, 255])
    lower_yellow = np.array([20, 80, 100])

    etude_video(balle_rouge, upper_red, lower_red)

