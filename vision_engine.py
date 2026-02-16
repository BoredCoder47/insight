import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import mediapipe as mp
import time
import numpy as np


class VisionEngine:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            use_gpu=False  # Force CPU mode
        )

        self.last_identity_check = 0
        self.identity_interval = 5

        self.looking_away_duration = 0
        self.last_gaze_time = time.time()

    def process_frame(self, frame):
        frame = cv2.resize(frame, (320, 240))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        face_present = False
        multiple_faces = False
        looking_away = False

        if results.multi_face_landmarks:
            face_present = True
            if len(results.multi_face_landmarks) > 1:
                multiple_faces = True

            landmarks = results.multi_face_landmarks[0]

            left_iris = landmarks.landmark[468]
            right_iris = landmarks.landmark[473]

            gaze_center_threshold = 0.02

            if abs(left_iris.x - right_iris.x) > gaze_center_threshold:
                looking_away = True

            if looking_away:
                self.looking_away_duration += 0.2
            else:
                self.looking_away_duration = 0

        return {
            "face_present": face_present,
            "multiple_faces": multiple_faces,
            "looking_away": looking_away,
            "looking_away_duration": self.looking_away_duration
        }
