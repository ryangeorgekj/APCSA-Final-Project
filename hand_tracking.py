import time
import cv2
import mediapipe as mp
import numpy as np
import simpleaudio as sa
import pygame

pygame.mixer.init(frequency=44100, size=-16, channels=1) # Advaith
pygame.mixer.set_num_channels(20)                   
                                                                            
def make_tone(frequency, duration=1.0, volume=0.3, sample_rate=44100):
    num_cycles = round(duration * frequency)      
    actual_duration = num_cycles / frequency
    
    num_samples = int(sample_rate * actual_duration)
    t = np.linspace(0, actual_duration, num_samples, False)
    note = np.sin(frequency * t * 2 * np.pi)
    audio = (note * (2**15 - 1) * volume).astype(np.int16)
    return pygame.sndarray.make_sound(audio)                                


thumb_sound  = make_tone(261.63)  # C4 - Do
index_sound  = make_tone(293.66)  # D4 - Re
middle_sound = make_tone(329.63)  # E4 - Mi
ring_sound   = make_tone(349.23)  # F4 - Fa
pinky_sound  = make_tone(392.00)  # G4 - Sol

# Advaith 

thumb_playing  = False
index_playing  = False
middle_playing = False
ring_playing   = False
pinky_playing  = False

thumb_channel  = None
index_channel  = None
middle_channel = None
ring_channel   = None
pinky_channel  = None

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Set up the hand landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1)

# Open webcam
cap = cv2.VideoCapture(0)
timestamp = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Flip horizontally so it acts like a mirror
        frame = cv2.flip(frame, 1)

        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hands
        timestamp += 33  # ~30 fps
        result = landmarker.detect_for_video(mp_image, timestamp)

        # Draw landmarks and connections
        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                # Draw each landmark as a green dot
                for lm in hand:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Draw connections between landmarks (skeleton)
                connections = [
                    (0,1),(1,2),(2,3),(3,4),         # thumb
                    (0,5),(5,6),(6,7),(7,8),         # index
                    (5,9),(9,10),(10,11),(11,12),    # middle
                    (9,13),(13,14),(14,15),(15,16),  # ring
                    (13,17),(17,18),(18,19),(19,20), # pinky
                    (0,17)                            # palm base
                ]
                # Ryan and a little bit of Advaith
                line = hand[9].y + 0.05
                # Thumb
                if hand[4].y > line:
                    cv2.putText(frame, "Do", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if not thumb_playing:
                        thumb_channel = thumb_sound.play(loops=-1)
                        thumb_playing = True
                else:
                    if thumb_playing:
                        thumb_channel.stop()
                        thumb_playing = False
                # Index
                if hand[8].y > line:
                    cv2.putText(frame, "Re", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if not index_playing:
                        index_channel = index_sound.play(loops=-1)
                        index_playing = True
                else:
                    if index_playing:
                        index_channel.stop()
                        index_playing = False
                # Middle
                if hand[12].y > line:
                    cv2.putText(frame, "Mi", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if not middle_playing:
                        middle_channel = middle_sound.play(loops=-1)
                        middle_playing = True
                else:
                    if middle_playing:
                        middle_channel.stop()
                        middle_playing = False
                # Ring
                if hand[16].y > line:
                    cv2.putText(frame, "Fa", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if not ring_playing:
                        ring_channel = ring_sound.play(loops=-1)
                        ring_playing = True
                else:
                    if ring_playing:
                        ring_channel.stop()
                        ring_playing = False
                # Pinky
                if hand[20].y > line:
                    cv2.putText(frame, "Sol", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if not pinky_playing:
                        pinky_channel = pinky_sound.play(loops=-1)
                        pinky_playing = True
                else:
                    if pinky_playing:
                        pinky_channel.stop()
                        pinky_playing = False

                # Ryan and a little bit of Advaith

                for start, end in connections:
                    x1 = int(hand[start].x * frame.shape[1])
                    y1 = int(hand[start].y * frame.shape[0])
                    x2 = int(hand[end].x * frame.shape[1])
                    y2 = int(hand[end].y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow('Hand Tracking - Press Q to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
