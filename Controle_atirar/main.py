#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medindo distância do dedo indicador até a palma da mão
"""

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import math

# Detecta tamanho da tela usando tkinter
root = tk.Tk()
root.withdraw()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Inicializa os módulos do MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configurações da detecção
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       model_complexity=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Captura da webcam
cap = cv2.VideoCapture(0)

cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Hand Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)


lim = [230,380]

cont = False

# Loop principal
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    black_frame = np.zeros_like(frame)
    
    
   
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Obtém pontos de interesse: 0 (palma) e 8 (ponta do indicador)
            h, w, _ = frame.shape
            palm_x = int(hand_landmarks.landmark[0].x * w)
            palm_y = int(hand_landmarks.landmark[0].y * h)
            finger_x = int(hand_landmarks.landmark[8].x * w)
            finger_y = int(hand_landmarks.landmark[8].y * h)

            # Calcula a distância euclidiana
            distance = math.hypot(finger_x - palm_x, finger_y - palm_y)
            
            

            # Mostra a distância na tela
            cv2.putText(black_frame, f'Distancia: {int(distance)} px', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            if distance > lim[1]:
                cont = True
            if cont == True and distance < lim[0]:     
                cv2.putText(black_frame, "ok", (400, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cont = False  # Reseta para detectar novo ciclo
            

            # Desenha os pontos de interesse
            cv2.circle(black_frame, (palm_x, palm_y), 10, (0, 255, 0), -1)    # Palma - verde
            cv2.circle(black_frame, (finger_x, finger_y), 10, (255, 0, 0), -1) # Indicador - azul
            cv2.line(black_frame, (palm_x, palm_y), (finger_x, finger_y), (255, 255, 0), 2)

            # Desenha toda a mão
            mp_drawing.draw_landmarks(black_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Exibe o resultado
    cv2.imshow('Hand Tracking', black_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
