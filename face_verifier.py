import numpy as np
from insightface.app import FaceAnalysis


class FaceVerifier:
    def __init__(self, threshold=0.65):
        self.threshold = threshold
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=-1)
        self.reference_embedding = None

    def register_reference(self, frame):
        faces = self.app.get(frame)

        if len(faces) != 1:
            return False

        self.reference_embedding = faces[0].embedding
        return True

    def analyze_frame(self, frame):
        faces = self.app.get(frame)

        face_present = len(faces) > 0
        multiple_faces = len(faces) > 1
        looking_away = False
        identity = None

        if len(faces) == 1:
            face = faces[0]

            # Identity check
            if self.reference_embedding is not None:
                score = self._cosine_similarity(
                    face.embedding,
                    self.reference_embedding
                )

                identity = {
                    "match": score >= self.threshold,
                    "score": float(score)
                }

            # Head turn detection (simple heuristic)
            left_eye = face.kps[0]
            right_eye = face.kps[1]
            nose = face.kps[2]

            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            eye_width = abs(right_eye[0] - left_eye[0])

            if eye_width > 0:
                ratio = (nose[0] - eye_center_x) / eye_width
                if abs(ratio) > 0.25:
                    looking_away = True

        return {
            "face_present": face_present,
            "multiple_faces": multiple_faces,
            "looking_away": looking_away,
            "identity": identity
        }

    def _cosine_similarity(self, a, b):
        return float(
            np.dot(a, b) /
            (np.linalg.norm(a) * np.linalg.norm(b))
        )
