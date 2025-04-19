#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 19:35:13 2025

@author: tercio
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import math
import tkinter as tk

# Detecta tamanho da tela usando tkinter
root = tk.Tk()
root.withdraw()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Inicializa os módulos do MediaPipe para detecção de mãos e desenho
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configurações para detecção de mãos
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       model_complexity=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Constantes
WIDTH, HEIGHT = screen_width, screen_height
CIRCLE_RADIUS = 20
BASE_SPEED = 5  # velocidade inicial das bolinhas

# Captura de vídeo da webcam
cap = cv2.VideoCapture(2)

cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Hand Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Lista para armazenar círculos
circles = []

# Contador de bolinhas capturadas
captured_count = 0

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Loop principal
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)  # Corrige o espelhamento horizontal

    # Converte o frame de BGR para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa o frame para detectar mãos
    results = hands.process(frame_rgb)

    # Cria um fundo preto com as mesmas dimensões do frame original
    black_frame = np.zeros_like(frame)

    # Lista de pontos das mãos
    hand_points = []

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            for landmark in hand_landmarks.landmark:
                px = int(landmark.x * frame.shape[1])
                py = int(landmark.y * frame.shape[0])
                hand_points.append((px, py))
            mp_drawing.draw_landmarks(black_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Atualiza a velocidade com base nas capturas
    CIRCLE_SPEED = BASE_SPEED + (captured_count // 10) * 5

    # Atualiza e desenha círculos
    new_circles = []
    for (x, y) in circles:
        y -= CIRCLE_SPEED  # move para cima

        # Verifica se o círculo colide com algum ponto da mão
        colidiu = False
        for (hx, hy) in hand_points:
            dist = math.hypot(x - hx, y - hy)
            if dist < CIRCLE_RADIUS:
                colidiu = True
                captured_count += 1
                break

        if not colidiu and y + CIRCLE_RADIUS > 0:
            new_circles.append((x, y))
            cv2.circle(black_frame, (x, y), CIRCLE_RADIUS, (0, 0, 255), -1)  # vermelho

    circles = new_circles

    # Adiciona novos círculos aleatoriamente
    if random.random() < 0.05:
        new_x = random.randint(CIRCLE_RADIUS, frame_width - CIRCLE_RADIUS)
        new_y = frame_height + CIRCLE_RADIUS
        circles.append((new_x, new_y))

    # Mostra o contador no canto superior direito
    text = f'Capturadas: {captured_count}'
    cv2.putText(black_frame, text, (frame_width - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Exibe o frame
    cv2.imshow('Hand Tracking', black_frame)

    # Fecha com tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
