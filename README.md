# APCSA-Final-Project
# Air Piano 🎹

An AP Computer Science A final project that turns your hand into a playable piano using computer vision. A webcam tracks your hand in real time, and curling each finger below your palm plays a musical note — no keyboard required.

## How It Works

The program uses Google's MediaPipe to detect 21 landmark points on your hand from the webcam feed. Each frame, it checks the vertical position of each fingertip against a "press line" placed just below the middle finger's knuckle. When a fingertip crosses below that line, it triggers a continuous tone for that finger; lifting the finger back up stops the tone. The notes are generated mathematically as sine waves rather than loaded from audio files.

## Notes

Each finger plays a note in the C major scale, labeled with its solfège syllable on screen:

| Finger | Solfège | Note | Frequency |
| ------ | ------- | ---- | --------- |
| Thumb  | Do      | C4   | 261.63 Hz |
| Index  | Re      | D4   | 293.66 Hz |
| Middle | Mi      | E4   | 329.63 Hz |
| Ring   | Fa      | F4   | 349.23 Hz |
| Pinky  | Sol     | G4   | 392.00 Hz |

Play them in order — thumb to pinky — for an ascending **Do Re Mi Fa Sol** scale.

## Requirements

- Python 3.9–3.12 (MediaPipe does not yet support 3.13+)
- A webcam

## Installation

Install the required libraries:

\`\`\`bash
pip3 install mediapipe opencv-contrib-python numpy pygame
\`\`\`

Then download the hand landmark model into the project folder:

\`\`\`bash
curl -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
\`\`\`

## Usage

Run the program from the project folder:

\`\`\`bash
python3 hand_tracking.py
\`\`\`

A window will open showing your webcam feed with your hand skeleton overlaid. Hold your hand up with fingers extended, then curl a finger down past the palm line to play its note. Press **Q** to quit.

## Tuning

A few values in the code can be adjusted to change how it feels:

- **Press sensitivity** — the `line = hand[9].y + 0.05` value sets how far below the knuckle a fingertip must go to trigger a note. Lower the `0.05` to make it more sensitive, raise it to require a bigger motion.
- **Volume** — the `volume` parameter in `make_tone` (default `0.3`) controls loudness.
- **Camera** — `cv2.VideoCapture(0)` selects the camera. Change `0` to `1` or `2` if the wrong camera is used (for example, when an iPhone connects as a Continuity Camera).

## Built With

- [MediaPipe](https://ai.google.dev/edge/mediapipe) — hand landmark detection
- [OpenCV](https://opencv.org/) — webcam capture and on-screen drawing
- [pygame](https://www.pygame.org/) — real-time audio playback
- [NumPy](https://numpy.org/) — sine-wave tone generation

## Authors

Ryan George and Advaith
