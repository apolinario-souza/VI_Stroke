#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jogo de Coleta de Maçãs - Versão Final Corrigida
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import tkinter as tk
import random
import os
from collections import deque
from datetime import datetime
import time
import pandas as pd
from constantes import lim, id_suj, web_cam, apple_speed, duracao,posicao_variavel,id_suj
 

# Configurações iniciais
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def load_image(path, default_color=None, alpha=False):
    if not os.path.exists(path):
        print(f"Aviso: Imagem não encontrada em {path}")
        if default_color:
            size = (100, 100, 4) if alpha else (100, 100, 3)
            dummy_img = np.zeros(size, dtype=np.uint8)
            dummy_img[:] = (*default_color, 255) if alpha else default_color
            return dummy_img
        return None
    
    try:
        flags = cv2.IMREAD_UNCHANGED if alpha else cv2.IMREAD_COLOR
        img = cv2.imread(path, flags)
        if img is None:
            raise ValueError("Não foi possível carregar a imagem")
            
        if alpha and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        elif not alpha and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img
        
    except Exception as e:
        print(f"Erro ao carregar imagem {path}: {str(e)}")
        if default_color:
            size = (100, 100, 4) if alpha else (100, 100, 3)
            dummy_img = np.zeros(size, dtype=np.uint8)
            dummy_img[:] = (*default_color, 255) if alpha else default_color
            return dummy_img
        return None

# Carrega as imagens
background_img = load_image('img/fundo.png')
basket_img = load_image('img/cesta.png', alpha=True)
apple_img = load_image('img/maca.png', alpha=True)

# Detecta tamanho da tela
root = tk.Tk()
root.withdraw()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Redimensiona as imagens
if background_img is not None:
    background_img = cv2.resize(background_img, (screen_width, screen_height))

# Configura o tamanho dos elementos
basket_height = int(screen_width*0.11)
apple_size = int(screen_width*0.05)

if basket_img is not None:
    aspect_ratio = basket_img.shape[1] / basket_img.shape[0]
    basket_width = int(basket_height * aspect_ratio)
    basket_img = cv2.resize(basket_img, (basket_width, basket_height))
else:
    basket_width = int(screen_width*0.11)

if apple_img is not None:
    aspect_ratio = apple_img.shape[1] / apple_img.shape[0]
    apple_width = int(apple_size * aspect_ratio)
    apple_img = cv2.resize(apple_img, (apple_width, apple_size))
else:
    apple_width = apple_size

# Configuração da captura de vídeo
cap = cv2.VideoCapture(web_cam)
cv2.namedWindow('Apple Catcher', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Apple Catcher', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Configurações do jogo
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Variáveis do jogo
history_size = 7
distance_history = deque(maxlen=history_size)
position_history = deque(maxlen=history_size)
MIN_DISTANCE = lim[0]
MAX_DISTANCE = lim[1]
basket_y = screen_height - basket_height - 50
track_width = int(screen_width * 0.8)
track_start = int((screen_width - track_width) / 2)
apples = []

spawn_rate = 45
spawn_counter = 0
pontos = 0
erros = 0
macas_perdidas = 0
acuracia = 0

class Apple:
    def __init__(self, x):
        self.x = x
        self.y = -apple_size
        self.collected = False
        self.rotation = random.randint(0, 360)
        self.rotation_speed = random.uniform(-3, 3)
    
    def update(self):
        self.y += apple_speed
        self.rotation += self.rotation_speed
        return self.y > screen_height + apple_size
    
    def draw(self, frame):
        if apple_img is not None and apple_img.shape[2] == 4:
            rot_mat = cv2.getRotationMatrix2D((apple_width//2, apple_size//2), self.rotation, 1)
            rotated_apple = cv2.warpAffine(apple_img, rot_mat, (apple_width, apple_size))
            
            y_start = max(0, int(self.y) - apple_size//2)
            y_end = min(frame.shape[0], int(self.y) + apple_size//2)
            x_start = max(0, int(self.x) - apple_width//2)
            x_end = min(frame.shape[1], int(self.x) + apple_width//2)
            
            if y_end > y_start and x_end > x_start:
                apple_y1 = max(0, apple_size//2 - int(self.y) if self.y < apple_size//2 else 0)
                apple_y2 = apple_y1 + (y_end - y_start)
                apple_x1 = max(0, apple_width//2 - int(self.x) if self.x < apple_width//2 else 0)
                apple_x2 = apple_x1 + (x_end - x_start)
                
                apple_roi = rotated_apple[apple_y1:apple_y2, apple_x1:apple_x2]
                
                if apple_roi.shape[0] > 0 and apple_roi.shape[1] > 0:
                    alpha = apple_roi[:, :, 3] / 255.0
                    for c in range(0, 3):
                        frame[y_start:y_end, x_start:x_end, c] = (
                            apple_roi[:, :, c] * alpha + 
                            frame[y_start:y_end, x_start:x_end, c] * (1.0 - alpha))
        else:
            cv2.circle(frame, (int(self.x), int(self.y)), apple_size//2, (0, 0, 255), -1)

spawn_order = 0  # Controla a ordem de spawn (0, 1, 2, 0, 1, 2...)

def spawn_apple():
    global spawn_order
    
    # Define as 3 posições x fixas na ordem desejada
    fixed_x_positions = [
        track_start + track_width // 4,    # 1/4
        track_start + 3 * track_width // 4,  # 3/4
        track_start + track_width // 2    # 1/2
    ]
    if posicao_variavel == 0:
        x = fixed_x_positions[spawn_order]
        spawn_order = (spawn_order + 1) % 3  # Cicla entre 0, 1, 2
    else:
        x = random.choice(fixed_x_positions)
        
    
    apples.append(Apple(x))
    


def draw_basket(frame, x, y):
    if basket_img is not None and basket_img.shape[2] == 4:
        y_start = max(0, y)
        y_end = min(frame.shape[0], y + basket_height)
        x_start = max(0, x)
        x_end = min(frame.shape[1], x + basket_width)
        
        if y_end > y_start and x_end > x_start:
            basket_y1 = max(0, -y)
            basket_y2 = basket_y1 + (y_end - y_start)
            basket_x1 = max(0, -x)
            basket_x2 = basket_x1 + (x_end - x_start)
            
            basket_roi = basket_img[basket_y1:basket_y2, basket_x1:basket_x2]
            alpha = basket_roi[:, :, 3] / 255.0
            
            for c in range(0, 3):
                frame[y_start:y_end, x_start:x_end, c] = (
                    basket_roi[:, :, c] * alpha + 
                    frame[y_start:y_end, x_start:x_end, c] * (1.0 - alpha))
    else:
        cv2.rectangle(frame, 
                     (x, y), 
                     (x + basket_width, y + basket_height), 
                     (0, 255, 0), -1)

tempo_total = duracao * 60
tempo_inicio = time.time()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if background_img is not None:
        game_frame = background_img.copy()
    else:
        game_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    spawn_counter += 1
    if spawn_counter >= spawn_rate:
        spawn_apple()
        spawn_counter = 0
    
    active_apples = []
    basket_x = track_start
    
    tempo_atual = time.time()
    tempo_decorrido = tempo_atual - tempo_inicio
    
    tempo_restante = max(0, int(tempo_total - tempo_decorrido))
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            
            wrist = hand_landmarks.landmark[0]
            fingertips = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]
            
            distances = [
                math.hypot((ft.x - wrist.x) * w, (ft.y - wrist.y) * h)
                for ft in fingertips
            ]
            
            avg_distance = np.mean(distances)
            distance_history.append(avg_distance)
            smoothed_distance = np.mean(distance_history)
            
            opening_degree = np.clip(
                (smoothed_distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE) * 100, 
                0, 100
            )
            
            target_basket_x = track_start + int((opening_degree / 100) * (track_width - basket_width))
            position_history.append(target_basket_x)
            basket_x = int(np.mean(position_history))
            
            temp_frame = game_frame.copy()
            
            
            cv2.addWeighted(temp_frame, 0.5, game_frame, 0.5, 0, game_frame)
    
    active_apples = []
    for apple in apples:
        if not apple.collected:
        # Verifica se a maçã foi coletada pela cesta
            if (basket_x < apple.x < basket_x + basket_width and
                basket_y < apple.y + apple_size//2 and
                apple.y - apple_size//2 < basket_y + basket_height):
                apple.collected = True
                pontos += 1
        
        # Se não foi coletada, verifica se saiu da tela
            if apple.update():  # Retorna True se a maçã saiu da parte inferior
                if not apple.collected:  # Só conta como perdida se não foi coletada
                    macas_perdidas += 1
            else:
                apple.draw(game_frame)
                active_apples.append(apple)
    apples = active_apples
    
    draw_basket(game_frame, basket_x, basket_y)
    
    
    
    
    
    cv2.putText(game_frame, f'Pontos: {pontos}', (screen_width - int(screen_width*0.2), int(screen_width*0.04)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
   
    if macas_perdidas == 0:
        acuracia = 0
        if pontos > 0:
           acuracia = pontos*100
    elif macas_perdidas > 0 and pontos >0:
        acuracia = pontos/macas_perdidas*100
    
        
     
    
    cv2.putText(game_frame, f'Acuracia: {acuracia:.1f}%', 
                (screen_width - 250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


    
    minutos = tempo_restante // 60
    segundos = tempo_restante % 60
    cv2.putText(game_frame, f'Tempo: {minutos:02d}:{segundos:02d}', (screen_width//2, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    cv2.imshow('Apple Catcher', game_frame)
    if cv2.waitKey(1) & 0xFF == 27 or tempo_restante ==0:
        break


data_hoje = datetime.today().strftime('%Y-%m-%d')
nome_arquivo = f'suj{id_suj}_{data_hoje}.xlsx'
df = pd.DataFrame([{
    'Pontos': pontos,
    'Macas_perdidas': macas_perdidas,
    'Precisao': acuracia
}])
df.to_excel('resultados/'+nome_arquivo, index=False)


cap.release()
cv2.destroyAllWindows()