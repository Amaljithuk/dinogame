import pygame
import cv2
import mediapipe as mp
import numpy as np
import asyncio
import platform

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino Game with Hand Gesture (MediaPipe)")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Dino settings
dino_x, dino_y = 50, 300
dino_width, dino_height = 40, 60
dino_jump_velocity = -15
dino_gravity = 0.8
dino_velocity = 0
dino_on_ground = True

# Obstacle settings
obstacle_width, obstacle_height = 20, 50
obstacle_speed = 5
obstacle_gap = 300
obstacles = []

# Game variables
score = 0
game_over = False
FPS = 60
clock = pygame.time.Clock()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Hand detection variable
hand_detected = False

def setup():
    global obstacles
    obstacles = []
    spawn_obstacle()

def spawn_obstacle():
    obstacles.append([WIDTH, 300])

def detect_hand_gesture(frame):
    global hand_detected
    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_detected = False
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get thumb and index finger tip coordinates
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Calculate distance between thumb and index finger tips
            distance = np.sqrt(
                (thumb_tip.x - index_tip.x) ** 2 +
                (thumb_tip.y - index_tip.y) ** 2
            )
            
            # Debug: Show distance
            cv2.putText(frame, f"Distance: {distance:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Open hand if distance is large (adjust threshold based on your setup)
            if distance > 0.2:  # Tune this threshold
                hand_detected = True
                
        # Display hand detection status
        cv2.putText(frame, f"Hand Detected: {hand_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

def update_loop():
    global dino_y, dino_velocity, dino_on_ground, score, game_over, obstacles, hand_detected
    
    # Read webcam frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        return
    
    # Detect hand gesture
    frame = detect_hand_gesture(frame)
    
    # Show webcam feed
    cv2.imshow('Hand Detection', frame)
    
    if not game_over:
        # Update dino
        if hand_detected and dino_on_ground:
            dino_velocity = dino_jump_velocity
            dino_on_ground = False

        dino_velocity += dino_gravity
        dino_y += dino_velocity

        if dino_y >= 300:
            dino_y = 300
            dino_velocity = 0
            dino_on_ground = True

        # Update obstacles
        new_obstacles = []
        for obs in obstacles:
            obs[0] -= obstacle_speed
            if obs[0] > -obstacle_width:
                new_obstacles.append(obs)
        obstacles = new_obstacles

        # Spawn new obstacle
        if not obstacles or obstacles[-1][0] < WIDTH - obstacle_gap:
            spawn_obstacle()

        # Collision detection
        dino_rect = pygame.Rect(dino_x, dino_y, dino_width, dino_height)
        for obs in obstacles:
            obs_rect = pygame.Rect(obs[0], obs[1], obstacle_width, obstacle_height)
            if dino_rect.colliderect(obs_rect):
                game_over = True

        # Update score
        score += 0.1

    # Draw
    screen.fill(WHITE)
    pygame.draw.rect(screen, BLACK, (dino_x, dino_y, dino_width, dino_height))
    for obs in obstacles:
        pygame.draw.rect(screen, BLACK, (obs[0], obs[1], obstacle_width, obstacle_height))
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {int(score)}", True, BLACK)
    screen.blit(score_text, (10, 10))
    hand_text = font.render(f"Hand Detected: {hand_detected}", True, BLACK)
    screen.blit(hand_text, (10, 50))
    if game_over:
        game_over_text = font.render("Game Over! Press R to Restart", True, BLACK)
        screen.blit(game_over_text, (WIDTH//2 - 150, HEIGHT//2))
    pygame.display.flip()

async def main():
    global score, game_over, dino_y, dino_velocity, dino_on_ground
    setup()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and game_over:
                score = 0
                game_over = False
                dino_y = 300
                dino_velocity = 0
                dino_on_ground = True
                setup()
        update_loop()
        # Check for 'q' to quit webcam
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            pygame.quit()
            return
        await asyncio.sleep(1.0 / FPS)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        try:
            asyncio.run(main())
        finally:
            cap.release()
            cv2.destroyAllWindows()
            pygame.quit()
#re_Nh6ExVwo_Borhv6Pa7DPDdtjhGde5Qiqy
#scrimba