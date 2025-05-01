import cv2
import mediapipe as mp
import numpy as np
import random
import math
import os
import tkinter as tk
from constantes import web_cam

# Inicializa o MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Inicializa captura
cap = cv2.VideoCapture(web_cam)

# Obtém o tamanho real da tela usando tkinter
root = tk.Tk()
root.withdraw()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Configura a janela para tela cheia
cv2.namedWindow('Space Impact', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Space Impact', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty('Space Impact', cv2.WND_PROP_TOPMOST, 1)

# Configura a captura de vídeo para o tamanho da tela
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Carrega imagens
def load_image(path, alpha=True):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED if alpha else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada em: {path}")
    return img

# Carrega imagens com tratamento de erro
try:
    nave_img = load_image(os.path.join('img', 'nave.png'))
    background_img = load_image(os.path.join('img', 'fundo.png'), alpha=False)
    asteroid_img = load_image(os.path.join('img', 'asteroide.png'))
    explosion_img = load_image(os.path.join('img', 'asteroide_explodido.png'))
except FileNotFoundError as e:
    print(e)
    exit()

# Redimensiona imagens
ship_width = int(screen_width * 0.05)
ship_height = int(ship_width * (nave_img.shape[0] / nave_img.shape[1]))
nave_img = cv2.resize(nave_img, (ship_width, ship_height))

asteroid_size = int(screen_width * 0.04)
asteroid_img = cv2.resize(asteroid_img, (asteroid_size * 2, asteroid_size * 2))
explosion_img = cv2.resize(explosion_img, (asteroid_size * 2, asteroid_size * 2))

# Prepara fundo em rolagem
bg_h, bg_w = background_img.shape[:2]
bg_ratio = bg_w / bg_h
screen_ratio = screen_width / screen_height

if bg_ratio > screen_ratio:
    new_width = int(bg_h * screen_ratio)
    start_x = (bg_w - new_width) // 2
    background_img = background_img[:, start_x:start_x + new_width]
else:
    new_height = int(bg_w / screen_ratio)
    start_y = (bg_h - new_height) // 2
    background_img = background_img[start_y:start_y + new_height, :]

background_img = cv2.resize(background_img, (screen_width, screen_height))
scroll_y = 0

def draw_image_with_alpha(background, img, x, y):
    h, w = img.shape[:2]
    x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
    x2, y2 = min(background.shape[1], x1 + w), min(background.shape[0], y1 + h)

    if x1 >= x2 or y1 >= y2:
        return

    img = img[:y2 - y1, :x2 - x1]
    overlay = background[y1:y2, x1:x2]

    if img.shape[2] == 4:
        alpha_img = img[:, :, 3] / 255.0
        alpha_bg = 1.0 - alpha_img

        for c in range(3):
            overlay[:, :, c] = (alpha_img * img[:, :, c] + alpha_bg * overlay[:, :, c])
    else:
        overlay[:] = img

    background[y1:y2, x1:x2] = overlay

# Inicializações do jogo
ship_x = screen_width // 2
ship_y = screen_height - ship_height - 20
bullets = []
asteroids = []
explosions = []
score = 0
hand_closed = False

# Loop do jogo
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # Move o fundo
    scroll_y = (scroll_y + 5) % screen_height
    game_frame = np.zeros_like(background_img)
    game_frame[0:screen_height - scroll_y] = background_img[scroll_y:]
    game_frame[screen_height - scroll_y:] = background_img[:scroll_y]

    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        lm = hand_landmarks[0]
        h, w = frame.shape[:2]

        wrist = lm.landmark[mp_hands.HandLandmark.WRIST]
        ship_x = int(wrist.x * screen_width)

        thumb_tip = lm.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        distance = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)

        if distance < 0.05 and not hand_closed:
            bullets.append([ship_x, ship_y])
            hand_closed = True
        elif distance >= 0.05:
            hand_closed = False

    ship_x = max(ship_width // 2, min(screen_width - ship_width // 2, ship_x))
    draw_image_with_alpha(game_frame, nave_img, ship_x, ship_y)

    new_bullets = []
    for bx, by in bullets:
        by -= int(screen_height * 0.03)
        if by > 0:
            new_bullets.append([bx, by])
            laser_length = int(screen_height * 0.03)
            cv2.line(game_frame, (bx, by), (bx, by - laser_length), (255, 255, 255), thickness=4)

            #cv2.circle(game_frame, (bx, by), int(screen_width * 0.01), (0, 255, 255), -1)
    bullets = new_bullets

    if random.random() < 0.03:
        ax = random.randint(asteroid_size, screen_width - asteroid_size)
        asteroids.append([ax, 0])

    new_asteroids = []
    for ax, ay in asteroids:
        ay += int(screen_height * 0.02)
        destroyed = False

        for i, (bx, by) in enumerate(bullets):
            if math.hypot(bx - ax, by - ay) < asteroid_size:
                destroyed = True
                score += 1
                bullets[i][1] = -100
                explosions.append([ax, ay, 10])
                break

        if abs(ax - ship_x) < ship_width // 2 and abs(ay - ship_y) < ship_height // 2:
            score = 0
            bullets = []
            asteroids = []
            explosions = []
            break

        if not destroyed and ay < screen_height:
            new_asteroids.append([ax, ay])
            draw_image_with_alpha(game_frame, asteroid_img, ax, ay)
    asteroids = new_asteroids

    # Exibir explosões
    new_explosions = []
    for ex, ey, frames_left in explosions:
        draw_image_with_alpha(game_frame, explosion_img, ex, ey)
        if frames_left > 1:
            new_explosions.append([ex, ey, frames_left - 1])
    explosions = new_explosions

    cv2.putText(game_frame, f'Pontos: {score}', (int(screen_width * 0.02), int(screen_height * 0.08)),
                cv2.FONT_HERSHEY_SIMPLEX, screen_width / 1000 * 1.5, (255, 255, 255), int(screen_width / 1000 * 3))

    cv2.imshow('Space Impact', game_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
