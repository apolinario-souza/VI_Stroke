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
import os
import time
import random
from constantes import web_cam, lim, velocidade_variavel, duracao, velocidade_mosca


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
cap = cv2.VideoCapture(web_cam)

cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Hand Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Carrega a imagem de fundo
background = cv2.imread(os.path.join('img', 'fundo.png'))
if background is not None:
    background = cv2.resize(background, (screen_width, screen_height))
    
# Carrega a imagem da mosca (substituindo a moeda)
mosca = cv2.imread(os.path.join('img', 'mosca.png'), cv2.IMREAD_UNCHANGED)
if mosca is not None:
    mosca = cv2.resize(mosca, (50, 50))  # Tamanho reduzido para a mosca
    if mosca.shape[2] == 4:
        mosca_bgr = mosca[:, :, :3]
        mosca_alpha = mosca[:, :, 3]
    else:
        mosca_bgr = mosca
        mosca_alpha = None

# Carrega e redimensiona as imagens da boca
def load_and_resize_boca(img_path, scale=0.2):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        img = cv2.resize(img, (width, height))
    return img

boca_aberta = load_and_resize_boca(os.path.join('img', 'boca_aberta.png'))
boca_fechada = load_and_resize_boca(os.path.join('img', 'boca_fechada.png'))

# Prepara as imagens da boca
def prepare_image(img):
    if img is not None:
        if img.shape[2] == 4:  # Se tiver canal alpha
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
        else:
            bgr = img
            alpha = None
        return bgr, alpha
    return None, None

boca_aberta_bgr, boca_aberta_alpha = prepare_image(boca_aberta)
boca_fechada_bgr, boca_fechada_alpha = prepare_image(boca_fechada)


cont = False
pontos = 0  # Variável para armazenar a pontuação
mosca_dentro = False  # Verifica se a mosca está dentro da boca
boca_aberta_anterior = False  # Estado anterior da boca
mostrar_mosca = True  # Controla quando a mosca deve ser mostrada
tempo_esconder_mosca = 0  # Tempo para manter a mosca escondida

# Variáveis para controlar a animação da mosca (horizontal)
mosca_x = -50  # Começa fora da tela (esquerda)
mosca_y = screen_height // 2  # Posição vertical fixa (meio da tela)



if velocidade_variavel == 1:
    mosca_speed = random.choice([15, 35, 55])  # Escolhe uma velocidade aleatória entre 30, 35, 40
else:
    mosca_speed = 35  # Define uma velocidade constante




tempo_total = duracao * 60
tempo_inicio = time.time()
cont_mosca = 0

# Loop principal
while cap.isOpened():
    print(cont_mosca)
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # Cria o frame com fundo preto ou com a imagem de fundo
    if background is not None:
        black_frame = background.copy()
    else:
        black_frame = np.zeros_like(frame)
    
    # Determina qual imagem da boca usar
    boca_img = None
    boca_alpha = None
    boca_aberta_atual = False  # Estado atual da boca
    
    
    tempo_atual = time.time()
    tempo_decorrido = tempo_atual - tempo_inicio
    
    tempo_restante = max(0, int(tempo_total - tempo_decorrido))
    
    
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
            
        
           
            # Determina se a mão está aberta ou fechada
            if distance > lim[1]:
                cont = True
                boca_img, boca_alpha = boca_aberta_bgr, boca_aberta_alpha
                boca_aberta_atual = True
            elif distance < lim[0]:
                cont = False
                boca_img, boca_alpha = boca_fechada_bgr, boca_fechada_alpha
                boca_aberta_atual = False
            else:
                # Mantém o estado anterior (aberto ou fechado)
                if cont:
                    boca_img, boca_alpha = boca_aberta_bgr, boca_aberta_alpha
                    boca_aberta_atual = True
                else:
                    boca_img, boca_alpha = boca_fechada_bgr, boca_fechada_alpha
                    boca_aberta_atual = False
            
            # Verifica se a boca estava aberta e agora fechou
            if boca_aberta_anterior and not boca_aberta_atual and mosca_dentro and mostrar_mosca:
                pontos += 1  # Incrementa a pontuação
                mostrar_mosca = False  # Esconde a mosca
                tempo_esconder_mosca = 15 # ~1 segundo (considerando ~30 FPS)
                cont_mosca+=1
                if velocidade_variavel == 1:
                    mosca_speed = random.choice(velocidade_mosca)  # Escolhe uma velocidade aleatória entre 30, 35, 40
                else:
                    mosca_speed = velocidade_mosca[1]  # Define uma velocidade constante
                
            
            boca_aberta_anterior = boca_aberta_atual  # Atualiza o estado anterior
            
            

         
    else:
        # Se nenhuma mão for detectada, mostra boca fechada
        boca_img, boca_alpha = boca_fechada_bgr, boca_fechada_alpha
        boca_aberta_atual = False
    
    # Adiciona a imagem da boca no centro da tela
    if boca_img is not None:
        # Calcula as coordenadas para centralizar a imagem
        img_height, img_width = boca_img.shape[:2]
        boca_x = int((screen_width - img_width) / 1.4)
        boca_y = int((screen_height - img_height) / 1.8)
        
        if boca_alpha is not None:
            # Aplica transparência se existir
            y1, y2 = boca_y, boca_y + img_height
            x1, x2 = boca_x, boca_x + img_width
            
            for c in range(0, 3):
                black_frame[y1:y2, x1:x2, c] = \
                    black_frame[y1:y2, x1:x2, c] * (1 - boca_alpha/255.0) + \
                    boca_img[:, :, c] * (boca_alpha/255.0)
        else:
            # Para imagens sem transparência
            black_frame[boca_y:boca_y+img_height, boca_x:boca_x+img_width] = boca_bgr
    
    # Controle do tempo para esconder a mosca
    if not mostrar_mosca:
        tempo_esconder_mosca -= 1
        if tempo_esconder_mosca <= 0:
            mostrar_mosca = True
            mosca_x = -50  # Reinicia a mosca do lado esquerdo
    
    # Atualiza a posição da mosca (movimento horizontal) se estiver visível
    if mostrar_mosca and mosca is not None:
        mosca_x += mosca_speed
        if mosca_x > screen_width:  # Se a mosca sair da tela, reinicia na esquerda
            mosca_x = -mosca.shape[1]
            cont_mosca+=1
            if velocidade_variavel == 1:
                mosca_speed = random.choice(velocidade_mosca)  # Escolhe uma velocidade aleatória entre 30, 35, 40
            else:
                mosca_speed = velocidade_mosca[1]  # Define uma velocidade constante
          
        
        y_offset = mosca_y - mosca.shape[0] // 2  # Centralizado verticalmente
        
        # Verifica se a mosca está dentro da boca
        if boca_img is not None:
            boca_center_x = boca_x + img_width // 2
            boca_center_y = boca_y + img_height // 2
            mosca_center_x = mosca_x + mosca.shape[1] // 2
            mosca_center_y = y_offset + mosca.shape[0] // 2
            
            # Distância entre o centro da mosca e o centro da boca
            dist = math.hypot(mosca_center_x - boca_center_x, mosca_center_y - boca_center_y)
            
            # Considera que a mosca está dentro se a distância for menor que um limiar
            mosca_dentro = dist < max(img_width, img_height) // 2
        else:
            mosca_dentro = False
        
        if mosca_alpha is not None:
            # Aplica transparência se existir
            y1, y2 = y_offset, y_offset + mosca.shape[0]
            x1, x2 = mosca_x, mosca_x + mosca.shape[1]
            
            if x1 >= 0 and x2 <= screen_width:  # Verifica se a mosca está dentro dos limites horizontais
                for c in range(0, 3):
                    black_frame[y1:y2, x1:x2, c] = \
                        black_frame[y1:y2, x1:x2, c] * (1 - mosca_alpha/255.0) + \
                        mosca_bgr[:, :, c] * (mosca_alpha/255.0)
        else:
            # Para imagens sem transparência
            if mosca_x >= 0 and mosca_x + mosca.shape[1] <= screen_width:
                black_frame[y_offset:y_offset+mosca.shape[0], mosca_x:mosca_x+mosca.shape[1]] = mosca_bgr
    
    # Mostra a pontuação na tela
    cv2.putText(black_frame, f'Pontos: {pontos}', (screen_width - 200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
       
    

    # Exibe o resultado
    cv2.imshow('Hand Tracking', black_frame)

    if cv2.waitKey(1) & 0xFF == 27 or tempo_restante ==0:
        break

cap.release()
cv2.destroyAllWindows()