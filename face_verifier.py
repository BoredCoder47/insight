import face_recognition
import numpy as np


class FaceVerifier:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reference_encoding = None

    def register_reference(self, frame):
        rgb = frame[:, :, ::-1]
        encodings = face_recognition.face_encodings(rgb)

        if len(encodings) != 1:
            return False

        self.reference_encoding = encodings[0]
        return True

    def analyze_frame(self, frame):
        rgb = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, face_locations)
        landmarks = face_recognition.face_landmarks(rgb)

        face_present = len(face_locations) > 0
        multiple_faces = len(face_locations) > 1
        looking_away = False
        identity = None

        if len(face_locations) == 1:
            # Identity check
            if self.reference_encoding is not None:
                distance = np.linalg.norm(
                    encodings[0] - self.reference_encoding
                )

                identity = {
                    "match": distance < self.threshold,
                    "score": float(1 - distance)
                }

            # Head turn detection (simple heuristic)
            if landmarks:
                left_eye = landmarks[0]["left_eye"]
                right_eye = landmarks[0]["right_eye"]
                nose_bridge = landmarks[0]["nose_bridge"]

                eye_center_x = (
                    np.mean([p[0] for p in left_eye]) +
                    np.mean([p[0] for p in right_eye])
                ) / 2

                eye_width = abs(
                    np.mean([p[0] for p in right_eye]) -
                    np.mean([p[0] for p in left_eye])
                )

                nose_x = np.mean([p[0] for p in nose_bridge])

                if eye_width > 0:
                    ratio = (nose_x - eye_center_x) / eye_width
                    if abs(ratio) > 0.35:
                        looking_away = True

        return {
            "face_present": face_present,
            "multiple_faces": multiple_faces,
            "looking_away": looking_away,
            "identity": identity
        }
