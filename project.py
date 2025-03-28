import math
import cv2 as cv
import numpy as np
from KalmanFilter import KalmanFilter

# --------- Affichage des métriques -------------
def affichage_video(frame, titre, param, param_pred, unite, x, y, decalY):
    cv.putText(frame, titre,
               (x, y - (decalY+40)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
               (255, 255, 255), 2)
    cv.putText(frame, f"{titre} {param:.2f} {unite}",
               (x, y - decalY), cv.FONT_HERSHEY_SIMPLEX, 0.5,
               (255, 255, 255), 2)
    cv.putText(frame, f"Prediction {titre} {param_pred:.2f} {unite}",
               (x, y - (decalY+20)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
               (255, 255, 255), 2)
    if param != 0:
        cv.putText(frame, f"ERREUR: {((param_pred-param)/abs(param))*100:.2f} %",
               (x, y - (decalY-20)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
               (31, 173, 255), 2)
    else:
        cv.putText(frame, "ERREUR: 0 %",
               (x, y - (decalY-20)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
               (31, 173, 255), 2)


# --------- Tracé de parabole -------------------

def trace_parabole(trajectoire):
    pts_parabole = None
    coeffs = None

    if len(trajectoire) > 2:
        listcX, listcY = [],[]
        for centroid in trajectoire:
            listcX.append(centroid[0])
            listcY.append(centroid[1])
        if len(listcX) > 2:
            coeffs = np.polyfit(listcX, listcY, 2)
        if coeffs is not None:
            poly = np.poly1d(coeffs)
            x_range = np.linspace(0, 1080, 100)
            y_range = poly(x_range)
            pts_parabole = np.array([x_range, y_range], dtype=np.int32).T

    return pts_parabole

# --------- Calcul distance ------------------
def calcul_distance(v0, angle, g, h):
    distance = (v0 / g) * math.cos(angle) * (
                v0 * math.sin(angle) + math.sqrt((v0 * math.sin(angle)) ** 2 + 2 * g * h))
    return distance


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
    v0_pred = 0
    xmax, ymax = 0, 0
    xmax_pred, ymax_pred = 0, 0
    area_min = 0
    angle_init = 0
    angle_init_pred = 0
    distance_totale = 0
    hauteur_max = 0
    distance_pred = distance_totale
    hauteur = 1 # Hauteur entre le sol et le bas du tableau en m
    trajectoire = []
    trajectoire_predite = []

    PIXEL_TO_METERS = 0.00185 # Coeff de proportionnalité entre les dimensions réelles et les dimensions utilisées dans la vidéo
    frame_counter = 0
    previous_centroid = None  # Dernière position connue de la balle
    previous_pred_centroid = None
    previous_frame_time = None  # Dernier timestamp
    g = 9.81  # Coefficient gravité
    compteur = 0
    compteur_v0 = 0
    compteur_pred = 0

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
        predicted_centroid = (int(predicted_state[0, 0]), int(predicted_state[1, 0]))  # Extraction de x et y prédits

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

            if video_path == "Video/Mousse_new.mp4":
                area_min = 1000
            elif video_path == "Video/rugby_tab.mp4":
                area_min = 5900
            elif video_path == "Video/rugby_tab.mp4":
                area_min = 300

            
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
            predicted_centroid = (int(predicted_state[0, 0]), int(predicted_state[1, 0]))
            cX_pred, cY_pred = predicted_centroid
            trajectoire_predite.append(predicted_centroid)
            trajectoire.append(centroid)



            # ----------- Calcul de l'angle initial de la balle + angle initial prédit -----------------
            if len(trajectoire) > 1 and len(trajectoire_predite) > 1:
                (x1, y1) = trajectoire[0]
                (x2, y2) = trajectoire[1]
                (x1_pred, y1_pred) = trajectoire_predite[0]
                (x2_pred, y2_pred) = trajectoire_predite[1]

                angle_init = - math.atan2((y2 - y1), (x2 - x1))
                angle_init_pred = - math.atan2((y2_pred - y1_pred), (x2_pred - x1_pred))

                x_prev, y_prev = previous_centroid
                x_current, y_current = centroid
                x_prev_pred, x_prev_pred = previous_pred_centroid
                x_current_pred, y_current_pred = predicted_centroid

                if y_prev < y_current and compteur == 0:
                    (xmax, ymax) = previous_centroid
                    compteur += 1

                if x_prev_pred < y_current_pred and compteur_pred == 0:
                    (xmax_pred, ymax_pred) = previous_pred_centroid
                    compteur_pred += 1


            # ------------ Calcul de l'angle et de la vitesse instantanée ------------------

            current_frame_time = frame_counter * frameTime
            if previous_centroid is not None and previous_pred_centroid is not None and previous_frame_time is not None:
                x_prev, y_prev = previous_centroid
                x_prev_pred, y_prev_pred = previous_pred_centroid
                delta_t = current_frame_time - previous_frame_time  # Temps en secondes
                if delta_t > 0:
                    # Pour la partie réelle
                    v_x = (x_prev - cX) / delta_t
                    v_y = (y_prev - cY) / delta_t
                    v_x = v_x * PIXEL_TO_METERS
                    v_y = v_y * PIXEL_TO_METERS
                    v = math.sqrt(v_x ** 2 + v_y ** 2)
                    # Pour la partie prédite
                    v_x_pred = (x_prev_pred - cX_pred) / delta_t
                    v_y_pred = (y_prev_pred - cY_pred) / delta_t
                    v_x_pred = v_x_pred * PIXEL_TO_METERS
                    v_y_pred = v_y_pred * PIXEL_TO_METERS
                    v_pred = math.sqrt(v_x_pred ** 2 + v_y_pred ** 2)

                    if compteur_v0 == 0:
                        v0 = v
                        v0_pred = v_pred
                        compteur_v0 += 1
                    cv.putText(transform, f"{v:.2f} m/s", (cX + 20, cY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv.putText(transform, f"{v_pred:.2f} m/s", (cX_pred + 20, cY_pred - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


            frame_counter += 1
            previous_centroid = centroid
            previous_pred_centroid = predicted_centroid
            previous_frame_time = current_frame_time

            future_trajectory_points = []

            # -------- Affichage de la position prédite (même si pas de détection) --------

        else:
            temp_kalman_filter = KalmanFilter(kalman_filter.dt, (kalman_filter.E[0, 0], kalman_filter.E[1, 0]),
                                              kalman_filter.pixel_to_meters)  # Créer une copie temporaire du filtre
            temp_kalman_filter.E = kalman_filter.E.copy()  # Copier l'état actuel
            temp_kalman_filter.P = kalman_filter.P.copy()  # Copier la covariance actuelle

            for _ in range(prediction_points_count):
                predicted_future_state = temp_kalman_filter.predict()
                predicted_future_position = (int(predicted_future_state[0, 0]), int(predicted_future_state[1, 0]))
                future_trajectory_points.append(predicted_future_position)


        if (xmax, ymax) != (0, 0):
            cv.line(transform, (xmax, ymax), (xmax, 621), (52, 235, 208), 1) # Hauteur max courbe réelle
        if (xmax_pred, ymax_pred) != (0, 0):
            cv.line(transform, (xmax_pred, ymax_pred), (xmax_pred, 621), (0, 0, 208), 1) # Hauteur max courbe prédite

        # -------- Tracé de parabole --------------
        parabole = trace_parabole(trajectoire)
        cv.polylines(transform, [parabole], False, (52, 235, 208), 1)

        for point in trajectoire_predite:
            cv.circle(transform, point, 3, (0, 0, 255), -1)  # Trajectoire en rouge

        for point in trajectoire:
            cv.circle(transform, point, 3, (52, 235, 208), -1)  # Trajectoire en jaune

        # -------- Affichage angle initial + vitesse initiale -------------
        affichage_video(transform,"ANGLE :",math.degrees(angle_init), math.degrees(angle_init_pred), "deg" , point_bg_new[0] + 20,point_bg_new[1] - 60,20)
        affichage_video(transform,"VITESSE INITIALE :",v0, v0_pred, "m/s", point_bg_new[0] + 400,point_bg_new[1] - 120,20)

        # -------- Affichage de la position prédite (temps réel) --------
        cv.circle(transform, predicted_centroid, 5, (0, 0, 255), -1) # En rouge

        # -------- Affichage de la trajectoire future prédite --------
        for future_point in future_trajectory_points:
            cv.circle(transform, future_point, 3, (255, 0, 0), -1) # En bleu


        # ---------- Calcul distance parcourue par la balle + distance prédite par Kalman ------------
        distance_totale = calcul_distance(v0, angle_init, g, hauteur)
        distance_pred = calcul_distance(v0_pred, angle_init_pred, g, hauteur)
        affichage_video(transform,"DISTANCE :",distance_totale, distance_pred, "m", point_bg_new[0] + 400,point_bg_new[1] - 20,20)


        cv.imshow('resultat', transform)
        cv.waitKey(20) & 0xFF
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

