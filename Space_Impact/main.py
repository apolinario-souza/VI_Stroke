import cv2
import mediapipe as mp
import numpy as np
import random
import math
import os

# Inicializa o MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Inicializa captura
cap = cv2.VideoCapture(2)
cv2.namedWindow('Space Impact', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Space Impact', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Tamanho da tela
screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Carrega imagem da nave
nave_img_path = os.path.join('img', 'nave.png')
nave_img = cv2.imread(nave_img_path, cv2.IMREAD_UNCHANGED)

if nave_img is None:
    raise FileNotFoundError(f"Imagem da nave não encontrada em: {nave_img_path}")

# Redimensiona a nave
ship_width, ship_height = 60, 60
nave_img = cv2.resize(nave_img, (ship_width, ship_height))

# Função para desenhar com canal alfa
def draw_image_with_alpha(background, img, x, y):
    h, w = img.shape[:2]
    x1, y1 = x - w // 2, y
    x2, y2 = x1 + w, y1 + h

    if x1 < 0 or y1 < 0 or x2 > background.shape[1] or y2 > background.shape[0]:
        return

    overlay = background[y1:y2, x1:x2]
    alpha_img = img[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_img

    for c in range(3):
        overlay[:, :, c] = (alpha_img * img[:, :, c] + alpha_bg * overlay[:, :, c])

    background[y1:y2, x1:x2] = overlay

# Inicializações do jogo
ship_x = screen_width // 2
ship_y = screen_height - ship_height - 20
bullets = []
asteroids = []
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

    # Fundo preto
    game_frame = np.zeros_like(frame)

    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        lm = hand_landmarks[0]
        h, w = frame.shape[:2]

        wrist = lm.landmark[mp_hands.HandLandmark.WRIST]
        ship_x = int(wrist.x * w)

        # Detecta se a mão está fechada com base na distância entre dedos
        thumb_tip = lm.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        distance = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)

        if distance < 0.05 and not hand_closed:
            bullets.append([ship_x, ship_y])
            hand_closed = True
        elif distance >= 0.05:
            hand_closed = False

    # Limita a posição da nave dentro da tela
    ship_x = max(ship_width // 2, min(screen_width - ship_width // 2, ship_x))

    # Desenha a nave
    draw_image_with_alpha(game_frame, nave_img, ship_x, ship_y)

    # Atualiza mísseis
    new_bullets = []
    for bx, by in bullets:
        by -= 15
        if by > 0:
            new_bullets.append([bx, by])
            cv2.circle(game_frame, (bx, by), 5, (0, 255, 255), -1)
    bullets = new_bullets

    # Gera asteroides
    if random.random() < 0.03:
        ax = random.randint(30, screen_width - 30)
        asteroids.append([ax, 0])

    # Atualiza asteroides
    new_asteroids = []
    for ax, ay in asteroids:
        ay += 7
        destroyed = False

        # Verifica colisão com mísseis
        for i, (bx, by) in enumerate(bullets):
            if math.hypot(bx - ax, by - ay) < 25:
                destroyed = True
                score += 1
                bullets[i][1] = -100  # remove o projétil
                break

        # Verifica colisão com a nave
        if abs(ax - ship_x) < ship_width // 2 and abs(ay - ship_y) < ship_height:
            score = 0
            bullets = []
            asteroids = []
            break

        if not destroyed and ay < screen_height:
            new_asteroids.append([ax, ay])
            cv2.circle(game_frame, (ax, ay), 25, (100, 100, 255), -1)
    asteroids = new_asteroids

    # Mostra pontuação
    cv2.putText(game_frame, f'Pontos: {score}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Exibe o jogo
    cv2.imshow('Space Impact', game_frame)

    # Pressione ESC para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
