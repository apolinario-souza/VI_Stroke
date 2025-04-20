#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medindo distância do dedo indicador até a palma da mão com suavização
"""

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import math
from collections import deque

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        
    def predict(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return int(predicted[0]), int(predicted[1])

# Detecta tamanho da tela
root = tk.Tk()
root.withdraw()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configurações com maior confiança
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      model_complexity=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)

# Filtros
kf_palm = KalmanFilter()
kf_finger = KalmanFilter()
history_size = 3
distance_history = deque(maxlen=history_size)

cap = cv2.VideoCapture(0)
cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Hand Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

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
            h, w, _ = frame.shape
            # Posições brutas
            raw_palm_x = int(hand_landmarks.landmark[0].x * w)
            raw_palm_y = int(hand_landmarks.landmark[0].y * h)
            raw_finger_x = int(hand_landmarks.landmark[8].x * w)
            raw_finger_y = int(hand_landmarks.landmark[8].y * h)
            
            # Aplicar Kalman Filter
            palm_x, palm_y = kf_palm.predict(raw_palm_x, raw_palm_y)
            finger_x, finger_y = kf_finger.predict(raw_finger_x, raw_finger_y)
            
            # Calcular distância
            distance = math.hypot(finger_x - palm_x, finger_y - palm_y)
            distance_history.append(distance)
            avg_distance = int(np.mean(distance_history)) if distance_history else 0
            
            # Mostrar informações
            cv2.putText(black_frame, f'Distancia: {avg_distance} px', (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # Desenhar elementos
            cv2.circle(black_frame, (palm_x, palm_y), 10, (0, 255, 0), -1)
            cv2.circle(black_frame, (finger_x, finger_y), 10, (255, 0, 0), -1)
            cv2.line(black_frame, (palm_x, palm_y), (finger_x, finger_y), (255, 255, 0), 2)
            mp_drawing.draw_landmarks(black_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Tracking', black_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()