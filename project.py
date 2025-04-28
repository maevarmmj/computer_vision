import math
import cv2 as cv
import numpy as np
from KalmanFilter import KalmanFilter
"""
Prérequis rapides : Soit mettre les 3 vidéos dans un dossier "Video",  
                    soit enlever "Video" dans les paths des fichiers (ex : "Video/Mousse.mp4"),
                    dans le if __name__ == "__main__"
"""

# --------- Affichage des métriques sur la vidéo -------------

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
        cv.putText(frame, f"ERREUR : {((param_pred-param)/abs(param))*100:.2f} %",
               (x, y - (decalY-20)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
               (31, 173, 255), 2)
    else:
        cv.putText(frame, "ERREUR : 0 %",
               (x, y - (decalY-20)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
               (31, 173, 255), 2)


# --------- Tracé de parabole -------------------

def trace_parabole(trajectoire):
    points_parabole = None
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
            points_parabole = np.array([x_range, y_range], dtype=np.int32).T

    return points_parabole

# --------- Calcul distance totale parcourue------------------

def calcul_distance(v0, angle, g, h):
    distance = (v0 / g) * math.cos(angle) * (
                v0 * math.sin(angle) + math.sqrt((v0 * math.sin(angle)) ** 2 + 2 * g * h))
    return distance


# --------- Calcul du moment pour la position du centroïde -------

def calculate_moment(mask):
    moments = cv.moments(mask)
    return moments

# --------- FONCTION PRINCIPALE ----------

def etude_video(balle):
    # --------- Initialisation des paramètres -----------------
    video_path = balle[0]
    upper_range = balle[1]
    lower_range = balle[2]

    cap = cv.VideoCapture(video_path)
    frameRate = 30.0  # Frame / s
    frameTime = 1.0 / frameRate
    PIXEL_TO_METERS = 0.00185 # Facteur d'échelle (px <-> m)
    hauteur = 1 # Hauteur entre le sol et le bas du tableau en m
    g = 9.81  # Coefficient gravité
    WIDTH = 1080
    HEIGHT = 621

    out = cv.VideoWriter('output.mp4', 0x7634706d, 20.0, (WIDTH, HEIGHT)) # Pour enregistrer la vidéo

    # Initialisation des vitesses initiales
    v0 = v0_pred =  0
    v = v_pred = 0
    angle_init = angle_init_pred = 0

    # Initialisation des positions max initiales
    x0 = x1 = y0 = y1 = x0_pred = x1_pred = y0_pred = y1_pred = xmax_pred = ymax_pred = xmax = ymax = 0

    # Initialisation de l'aire minimale de chaque balle
    area_min = 0

    # Initialisation des tableaux
    trajectoire = []
    trajectoire_predite = []
    future_trajectory_points = []
    trajectoire_v = [] # Stocke le temps + position y à chaque frame

    previous_centroid = None  # Dernière position connue de la balle
    previous_pred_centroid = None
    previous_frame_time = None  # Dernier timestamp
    predicted_centroid = None

    # Initialisation de compteurs variés
    compteur = 0
    compteur_v0 = 0
    compteur_pred = 0
    frame_counter = 0

    # -------- Initialisation du filtre de Kalman --------
    predicted_position = (0, 0)
    kalman_filter = KalmanFilter(frameTime, predicted_position, PIXEL_TO_METERS)
    prediction_points_count = 30  # Prédiction de x points après la fin de détection de la balle


    # -------- BOUCLE PRINCIPALE --------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fin de la vidéo...")
            break
        frame = cv.resize(frame, (0,0), fx=0.7, fy=0.7)

        # ------------- Partie sur l'homographie -----------------
        
        point_hg = (125, 10)
        point_hd = (685, 332)
        point_bg = (25, 522)
        point_bd = (672, 590)
        pts1 = np.float32([point_hg, point_hd, point_bg, point_bd]) # Points du tableau sur la video

        point_hg_new = (0, 0)
        point_hd_new = (WIDTH, 0)
        point_bg_new = (0, HEIGHT)
        point_bd_new = (WIDTH, HEIGHT)
        pts2 = np.float32([point_hg_new, point_hd_new, point_bg_new, point_bd_new]) # Points finaux

        M = cv.getPerspectiveTransform(pts1, pts2)

        transform = cv.warpPerspective(frame, M, (WIDTH, HEIGHT)) # Facteur 1,85 entre les dim réelles et finales (-> PIXEL_TO_METERS)

        # ---------- Partie détection de l'objet -------------
        hsv = cv.cvtColor(transform, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_range, upper_range)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


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

            if video_path == "Video/Mousse.mp4":
                area_min = 200
            elif video_path == "Video/Rugby.mp4":
                area_min = 5600
            elif video_path == "Video/Tennis.mp4":
                area_min = 30

            if area > area_min:

                # ----------- Calcul du centroïde ---------------------
                moment_mask = calculate_moment(mask)
                cX = int(moment_mask["m10"] / moment_mask["m00"])
                cY = int(moment_mask["m01"] / moment_mask["m00"])
                centroid = (cX,cY)
                cv.circle(transform, centroid, 5, (52, 235, 208), -1)


                # -------- Mise à jour du filtre de Kalman avec la détection --------
                measurement = np.matrix([[cX], [cY]]) # Mesure = centroïde détecté (ne sert que lorsqu'un nouveau centroïde est détecté pour adapter la mesure du filtre)
                kalman_filter.update(measurement)

                # -------- Mise à jour de la position prédite  ----------
                predicted_state = kalman_filter.predict() # Prédiction après l'update pour avoir la position la plus à jour
                predicted_centroid = (int(predicted_state[0, 0]), int(predicted_state[1, 0]))
                trajectoire_predite.append(predicted_centroid)

                # ----------- Calcul de l'angle initial de la balle + hauteur max de la courbe -----------------
                if len(trajectoire) >1 and len(trajectoire_predite) >1:
                    (x0, y0) = trajectoire[0]
                    (x1, y1) = trajectoire[1]
                    (x0_pred, y0_pred) = trajectoire_predite[0]
                    (x1_pred, y1_pred) = trajectoire_predite[1]

                    angle_init = - math.atan2((y1 - y0), (x1 - x0))
                    angle_init_pred = - math.atan2((y1_pred - y0_pred), (x1_pred - x0_pred))

                    x_prev, y_prev = previous_centroid
                    x_current, y_current = centroid
                    x_prev_pred, y_prev_pred = previous_pred_centroid
                    x_current_pred, y_current_pred = predicted_centroid

                    if y_prev < y_current and compteur == 0:
                        (xmax, ymax) = previous_centroid
                        compteur += 1

                    if y_prev_pred < y_current_pred and compteur_pred == 0:
                        (xmax_pred, ymax_pred) = previous_pred_centroid
                        compteur_pred += 1


                # ------------ Calcul de la vitesse initiale + instantanée ------------------

                current_frame_time = frame_counter * frameTime
                if previous_centroid is not None and previous_pred_centroid is not None and previous_frame_time is not None:
                    x_prev, y_prev = previous_centroid
                    delta_t = current_frame_time - previous_frame_time  # Temps en secondes
                    if delta_t > 0:
                        # Pour la partie réelle
                        v_x = (x_prev - cX) / delta_t
                        v_y = (y_prev - cY) / delta_t
                        v_x = v_x * PIXEL_TO_METERS
                        v_y = v_y * PIXEL_TO_METERS
                        v = math.sqrt(v_x ** 2 + v_y ** 2)

                        # Pour la partie prédite
                        v_x_pred = predicted_state[2, 0]
                        v_y_pred = predicted_state[3, 0]
                        v_x_pred = v_x_pred * PIXEL_TO_METERS
                        v_y_pred = v_y_pred * PIXEL_TO_METERS
                        v_pred = math.sqrt(v_x_pred ** 2 + v_y_pred ** 2)

                        if compteur_v0 == 0:
                            v0 = v
                            v0_pred = v_pred
                            compteur_v0 += 1

                trajectoire.append(centroid)
                trajectoire_v.append((current_frame_time, cY))

                previous_centroid = centroid
                previous_pred_centroid = predicted_centroid
                previous_frame_time = current_frame_time

            future_trajectory_points = []
            frame_counter += 1


        elif frame_counter > 5:
            # -------- Prédiction de la position même sans détection (mais après que le frame_counter soit > 5
            # -------- pour éviter toute détection AVANT apparition de la balle --------

            predicted_state = kalman_filter.predict()
            predicted_centroid = (int(predicted_state[0, 0]), int(predicted_state[1, 0]))

            # Afficher la vitesse prédite même après arrêt de détection de la balle
            v_x_pred = predicted_state[2, 0]
            v_y_pred = predicted_state[3, 0]
            v_x_pred = v_x_pred * PIXEL_TO_METERS
            v_y_pred = v_y_pred * PIXEL_TO_METERS
            v_pred = math.sqrt(v_x_pred ** 2 + v_y_pred ** 2)

            # -------- Calcul de la trajectoire future (points bleus foncés) --------
            temp_kalman_filter = KalmanFilter(kalman_filter.dt, (kalman_filter.E[0, 0], kalman_filter.E[1, 0]),
                                              kalman_filter.pixel_to_meters) # Copie temporaire
            temp_kalman_filter.E = kalman_filter.E.copy()
            temp_kalman_filter.P = kalman_filter.P.copy()

            for _ in range(prediction_points_count):
                predicted_future_state = temp_kalman_filter.predict()
                predicted_future_position = (int(predicted_future_state[0, 0]), int(predicted_future_state[1, 0]))
                future_trajectory_points.append(predicted_future_position)

            frame_counter += 1

        # -------- Tracé de la hauteur max (juste pour la comparaison) SANS PRENDRE EN COMPTE LE BAS DU TABLEAU  --------------
        if (xmax, ymax) != (0, 0):
            cv.arrowedLine(transform, (xmax, y0), (xmax, ymax + 10), (52, 235, 208), 2, tipLength=0.1)
            cv.line(transform, (x0, y0), (xmax, y0), (52, 235, 208), 1)

        if (xmax_pred, ymax_pred) != (0, 0):
            cv.arrowedLine(transform, (xmax_pred, y0_pred), (xmax_pred, ymax_pred), (255, 255, 0), 2, tipLength=0.1) # Hauteur max courbe prédite
            cv.line(transform, (x0_pred, y0_pred), (xmax_pred, y0_pred), (255, 255, 0), 1)

            y_0_ball = (HEIGHT-y0)
            y_0_ball_pred = (HEIGHT - y0_pred)
            hauteur_max = (HEIGHT-ymax-y_0_ball)*PIXEL_TO_METERS
            hauteur_max_pred = (HEIGHT-ymax-y_0_ball_pred)*PIXEL_TO_METERS
            affichage_video(transform, "HAUTEUR MAX :", hauteur_max, hauteur_max_pred, "m",
                    point_hd_new[0] - 800, point_hd_new[1]+100, 20)

        # ----------- Tracé de parabole -------------------
        parabole = trace_parabole(trajectoire)
        cv.polylines(transform, [parabole], False, (52, 235, 208), 1)
        for point in trajectoire:
            cv.circle(transform, point, 4, (52, 235, 208), -1)  # Trajectoire en jaune
        for point in trajectoire_predite:
            cv.circle(transform, point, 4, (255, 255, 0), -1)  # Trajectoire en cyan


        cv.putText(transform, f"Vitesse mesuree : {v:.2f} m/s", (point_bg_new[0] + 20, point_bg_new[1] - 140), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv.putText(transform, f"Vitesse predite : {v_pred:.2f} m/s", (point_bg_new[0] + 20, point_bg_new[1] - 120), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 0), 2)

        # -------- Affichage angle initial + vitesse initiale -------------
        affichage_video(transform,"ANGLE :",math.degrees(angle_init), math.degrees(angle_init_pred), "deg" , point_bg_new[0] + 20, point_bg_new[1] - 20,20)
        affichage_video(transform,"VITESSE INITIALE :",v0, v0_pred, "m/s", point_bg_new[0] + 400, point_bg_new[1] - 120,20)

        # -------- Affichage de la trajectoire future prédite --------
        for future_point in future_trajectory_points:
            cv.circle(transform, future_point, 4, (255, 0, 0), -1) # En bleu foncé

        # -------- Affichage de la position prédite (temps réel) --------
        cv.circle(transform, predicted_centroid, 4, (255, 255, 0), -1) # En cyan

        # ---------- Calcul distance parcourue par la balle + distance prédite par Kalman ------------
        distance_totale = 0
        distance_pred = 0
        if len(trajectoire) > 1:
            hauteur_0 = hauteur + (HEIGHT-trajectoire[0][1])*PIXEL_TO_METERS # Distance sol - balle
            distance_totale = calcul_distance(v0, angle_init, g, hauteur_0)
            distance_pred = calcul_distance(v0_pred, angle_init_pred, g, hauteur_0)
        affichage_video(transform,"DISTANCE :",distance_totale, distance_pred, "m", point_bg_new[0] + 400,point_bg_new[1] - 20,20)

        cv.imshow('resultat', transform)
        out.write(transform)
        cv.waitKey(5) & 0xFF
        if cv.waitKey(int(frameTime * 1000)) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

    # ----------- PARTIE POUR DETERMINER g --------------
    timestamps = np.array([item[0] for item in trajectoire_v])
    pixels_y = np.array([item[1] for item in trajectoire_v])

    y_meters = (HEIGHT - pixels_y) * PIXEL_TO_METERS # Adaptation au repère physique (conversion px -> m)
    # Modèle: y(t) = A*t^2 + B*t + C
    # Correspondance: A = -0.5*g, B = v0y, C = y0
    coeffs_y = np.polyfit(timestamps, y_meters, 2)
    A = coeffs_y[0]
    B = coeffs_y[1]
    C = coeffs_y[2]

    g_estimated = -2 * A

    print(f"y(t) = At^2 + Bt + C:")
    print(f"  A (-0.5*g) = {A:.4f}")
    print(f"  B (v0y)   = {B:.4f}")
    print(f"  C (h0) = {C:.4f}")
    print(f"  => g estimé = -2 * A = {g_estimated:.2f} m/s²")
    hauteur_0 = hauteur + (HEIGHT - C) * PIXEL_TO_METERS  # Distance sol - balle
    print(f"Distance parcourue estimée après détermination de g et h0 : {calcul_distance(v0, angle_init, g_estimated, hauteur_0):.2f}")

if __name__ == "__main__":
    # BALLE ROUGE
    balle_rouge = "Video/Mousse.mp4"
    upper_red = np.array([255, 255, 246])
    lower_red = np.array([0, 135, 22])

    # BALLE DE RUGBY
    balle_rugby = "Video/Rugby.mp4"
    upper_rugby = np.array([110, 255, 120])
    lower_rugby = np.array([90, 200, 30])

    # BALLE JAUNE
    balle_jaune = "Video/Tennis.mp4"
    upper_yellow = np.array([40, 255, 255])
    lower_yellow = np.array([20, 80, 100])

    red = [balle_rouge, upper_red, lower_red]
    rugby = [balle_rugby, upper_rugby, lower_rugby]
    yellow = [balle_jaune, upper_yellow, lower_yellow]

    # Choix entre "red", "yellow", "rugby",
    etude_video(yellow)
