#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jogo Completo: Caça aos Balões com Pedras e Efeitos de Explosão
"""

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import math
import os
import time
import random
import pandas as pd
from datetime import datetime
from constantes import web_cam, lim, velocidade_variavel, duracao,pedra_speed, velocidade_balao, id_suj

# Detecta tamanho da tela
root = tk.Tk()
root.withdraw()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       model_complexity=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Configuração da webcam
cap = cv2.VideoCapture(web_cam)
cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Hand Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Funções auxiliares para carregar imagens
def load_image(path, size=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None and size:
        img = cv2.resize(img, size)
    return img

def prepare_image(img):
    if img is not None:
        if img.shape[2] == 4:
            return img[:, :, :3], img[:, :, 3]
        return img, None
    return None, None

# Carrega todas as imagens
background = load_image(os.path.join('img', 'fundo.png'), (screen_width, screen_height))
balao = load_image(os.path.join('img', 'zumbi.png'), (int(screen_width*0.12), int(screen_width*0.12)))
pedra_img = load_image(os.path.join('img', 'bala.png'), (int(screen_width*0.05), int(screen_width*0.05)))
balao_explodindo_img = load_image(os.path.join('img', 'zumbi_atingido.png'), (int(screen_width*0.13), int(screen_width*0.13)))

# Prepara imagens com transparência
balao_bgr, balao_alpha = prepare_image(balao)
pedra_bgr, pedra_alpha = prepare_image(pedra_img)
balao_explodindo_bgr, balao_explodindo_alpha = prepare_image(balao_explodindo_img)

# Carrega imagens da boca
def load_boca(img_path, scale=0.2):
    img = load_image(img_path)
    if img is not None:
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        return cv2.resize(img, (width, height))
    return None

boca_aberta = load_boca(os.path.join('img', 'arma_disparo.png'))
boca_fechada = load_boca(os.path.join('img', 'arma.png'))
boca_aberta_bgr, boca_aberta_alpha = prepare_image(boca_aberta)
boca_fechada_bgr, boca_fechada_alpha = prepare_image(boca_fechada)

# Variáveis do jogo
cont = False
pontos = 0
balao_dentro = False
boca_aberta_anterior = False
mostrar_balao = True
tempo_esconder_balao = 0
balao_x = -50
balao_y = screen_height//2
pedras = []
cont_balao = 0
acertos_pedra = 0
tempo_inicio = time.time()
tempo_total = duracao * 60
explosao_ativa = False
explosao_tempo = 0
explosao_pos = (0, 0)
explosao_duracao = 10

if velocidade_variavel == 1:
    balao_speed = random.choice(velocidade_balao)
else:
    balao_speed = velocidade_balao[1]

# Loop principal
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    black_frame = background.copy() if background is not None else np.zeros_like(frame)
    
    # Controle da boca
    boca_img, boca_alpha = None, None
    boca_aberta_atual = False
    
    tempo_atual = time.time()
    tempo_decorrido = tempo_atual - tempo_inicio
    tempo_restante = max(0, int(tempo_total - tempo_decorrido))
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            palm_x = int(hand_landmarks.landmark[0].x * w)
            palm_y = int(hand_landmarks.landmark[0].y * h)
            finger_x = int(hand_landmarks.landmark[8].x * w)
            finger_y = int(hand_landmarks.landmark[8].y * h)

            distance = math.hypot(finger_x - palm_x, finger_y - palm_y)
            
            if distance > lim[1]:
                cont = True
                boca_img, boca_alpha = boca_aberta_bgr, boca_aberta_alpha
                boca_aberta_atual = True
            elif distance < lim[0]:
                cont = False
                boca_img, boca_alpha = boca_fechada_bgr, boca_fechada_alpha
                boca_aberta_atual = False
            else:
                if cont:
                    boca_img, boca_alpha = boca_aberta_bgr, boca_aberta_alpha
                    boca_aberta_atual = True
                else:
                    boca_img, boca_alpha = boca_fechada_bgr, boca_fechada_alpha
                    boca_aberta_atual = False
            
            # Lança pedra quando a boca abre
            if not boca_aberta_anterior and boca_aberta_atual:
                boca_center_x = (screen_width - boca_img.shape[1]) // 2 + boca_img.shape[1] // 2
                boca_center_y = (screen_height - boca_img.shape[0]) + boca_img.shape[0] // 2
                pedras.append({'x': boca_center_x, 'y': boca_center_y})
            
            # Pega balão quando a boca fecha
            if boca_aberta_anterior and not boca_aberta_atual and balao_dentro and mostrar_balao:
                pontos += 1
                mostrar_balao = False
                tempo_esconder_balao = 7
                cont_balao += 1
               
            boca_aberta_anterior = boca_aberta_atual
    else:
        boca_img, boca_alpha = boca_fechada_bgr, boca_fechada_alpha
        boca_aberta_atual = False
    
    # Desenha a boca
    if boca_img is not None:
        img_height, img_width = boca_img.shape[:2]
        boca_x = (screen_width - img_width) // 2
        boca_y = (screen_height - img_height)
        
        if boca_alpha is not None:
            y1, y2 = boca_y, boca_y + img_height
            x1, x2 = boca_x, boca_x + img_width
            
            for c in range(0, 3):
                black_frame[y1:y2, x1:x2, c] = \
                    black_frame[y1:y2, x1:x2, c] * (1 - boca_alpha/255.0) + \
                    boca_img[:, :, c] * (boca_alpha/255.0)
        else:
            black_frame[boca_y:boca_y+img_height, boca_x:boca_x+img_width] = boca_bgr
    
    # Atualiza e desenha pedras
    for pedra in pedras[:]:
        pedra['y'] -= pedra_speed
        
        # Verifica colisão com balão
        if mostrar_balao and balao is not None:
            pedra_rect = {
                'x1': pedra['x'] - 25,
                'y1': pedra['y'] - 25,
                'x2': pedra['x'] + 25,
                'y2': pedra['y'] + 25
            }
            
            balao_rect = {
                'x1': balao_x,
                'y1': balao_y - balao.shape[0]//2,
                'x2': balao_x + balao.shape[1],
                'y2': balao_y + balao.shape[0]//2
            }
            
            if (pedra_rect['x1'] < balao_rect['x2'] and
                pedra_rect['x2'] > balao_rect['x1'] and
                pedra_rect['y1'] < balao_rect['y2'] and
                pedra_rect['y2'] > balao_rect['y1']):
                
                pontos += 1
                acertos_pedra += 1
                mostrar_balao = False
                tempo_esconder_balao = 15
                cont_balao += 1
                pedras.remove(pedra)
                balao_speed = random.choice(velocidade_balao) if velocidade_variavel == 1 else velocidade_balao[1]
                
                explosao_ativa = True
                explosao_tempo = explosao_duracao
                explosao_pos = (balao_rect['x1'] + balao.shape[1]//2 - 60,
                               balao_rect['y1'] + balao.shape[0]//2 - 60)
                continue
        
        # Remove pedras que saíram da tela
        if pedra['y'] < -50:
            pedras.remove(pedra)
        else:
            # Desenha pedra
            if pedra_img is not None:
                x = pedra['x'] - pedra_img.shape[1]//2
                y = pedra['y'] - pedra_img.shape[0]//2
                
                if pedra_alpha is not None:
                    y1, y2 = max(0, y), min(screen_height, y + pedra_img.shape[0])
                    x1, x2 = max(0, x), min(screen_width, x + pedra_img.shape[1])
                    
                    if y1 < y2 and x1 < x2:
                        alpha_s = pedra_alpha[y1-y:y2-y, x1-x:x2-x] / 255.0
                        alpha_l = 1.0 - alpha_s
                        
                        for c in range(0, 3):
                            black_frame[y1:y2, x1:x2, c] = \
                                black_frame[y1:y2, x1:x2, c] * alpha_l + \
                                pedra_bgr[y1-y:y2-y, x1-x:x2-x, c] * alpha_s
    
    # Mostra explosão se ativa
    if explosao_ativa:
        if balao_explodindo_img is not None:
            x, y = explosao_pos
            if balao_explodindo_alpha is not None:
                y1, y2 = max(0, y), min(screen_height, y + balao_explodindo_img.shape[0])
                x1, x2 = max(0, x), min(screen_width, x + balao_explodindo_img.shape[1])
                
                if y1 < y2 and x1 < x2:
                    alpha_s = balao_explodindo_alpha[y1-y:y2-y, x1-x:x2-x] / 255.0
                    alpha_l = 1.0 - alpha_s
                    
                    for c in range(0, 3):
                        black_frame[y1:y2, x1:x2, c] = \
                            black_frame[y1:y2, x1:x2, c] * alpha_l + \
                            balao_explodindo_bgr[y1-y:y2-y, x1-x:x2-x, c] * alpha_s
        
        explosao_tempo -= 1
        if explosao_tempo <= 0:
            explosao_ativa = False
    
    # Controle do balão
    if not mostrar_balao:
        tempo_esconder_balao -= 1
        if tempo_esconder_balao <= 0:
            mostrar_balao = True
            balao_x = -50
            balao_y = screen_height//2
    
    if mostrar_balao and balao is not None:
        balao_x += balao_speed
        if balao_x > screen_width:
            balao_x = -balao.shape[1]
            cont_balao += 1
            balao_y = screen_height//2
            balao_speed = random.choice(velocidade_balao) if velocidade_variavel == 1 else velocidade_balao[1]
        
        y_offset = balao_y - balao.shape[0]//2
        
        # Verifica se balão está na boca
        if boca_img is not None:
            boca_center_x = (screen_width - boca_img.shape[1])//2 + boca_img.shape[1]//2
            boca_center_y = (screen_height - boca_img.shape[0]) + boca_img.shape[0]//2
            balao_center_x = balao_x + balao.shape[1]//2
            balao_center_y = y_offset + balao.shape[0]//2
            
            dist = math.hypot(balao_center_x - boca_center_x, balao_center_y - boca_center_y)
            balao_dentro = dist < max(boca_img.shape[0], boca_img.shape[1])//2
        
        # Desenha balão
        if balao_alpha is not None:
            y1, y2 = y_offset, y_offset + balao.shape[0]
            x1, x2 = balao_x, balao_x + balao.shape[1]
            
            if x1 >= 0 and x2 <= screen_width:
                for c in range(0, 3):
                    black_frame[y1:y2, x1:x2, c] = \
                        black_frame[y1:y2, x1:x2, c] * (1 - balao_alpha/255.0) + \
                        balao_bgr[:, :, c] * (balao_alpha/255.0)
    
    # Mostra informações
    cv2.putText(black_frame, f'Pontos: {pontos}', (screen_width - 250, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(black_frame, f'Acuracia: {(acertos_pedra/cont_balao*100 if cont_balao > 0 else 0):.1f}%', 
                (screen_width - 250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    minutos = tempo_restante // 60
    segundos = tempo_restante % 60
    cv2.putText(black_frame, f'Tempo: {minutos:02d}:{segundos:02d}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    cv2.imshow('Hand Tracking', black_frame)

    if cv2.waitKey(1) & 0xFF == 27 or tempo_restante == 0:
        break

# Salva resultados
data_hoje = datetime.today().strftime('%Y-%m-%d')
nome_arquivo = f'suj{id_suj}_{data_hoje}.xlsx'
df = pd.DataFrame([{
    'Pontos': pontos,
    'n_balaos': cont_balao,
    'precisao': f"{(acertos_pedra/cont_balao*100 if cont_balao > 0 else 0):.1f}%"
}])
df.to_excel('resultados/'+nome_arquivo, index=False)

cap.release()
cv2.destroyAllWindows()