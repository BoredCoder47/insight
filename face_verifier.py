import numpy as np
import cv2
from insightface.app import FaceAnalysis


class FaceVerifier:
    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=-1)
        self.reference_embedding = None

    def register_reference(self, frame):
        emb = self._get_embedding(frame)
        if emb is not None:
            self.reference_embedding = emb
            return True
        return False

    def verify(self, frame):
        if self.reference_embedding is None:
            return None

        emb = self._get_embedding(frame)
        if emb is None:
            return {"match": False, "score": 0}

        score = self._cosine_similarity(emb, self.reference_embedding)
        return {
            "match": score >= self.threshold,
            "score": score
        }

    def _get_embedding(self, frame):
        faces = self.app.get(frame)
        if not faces:
            return None
        return faces[0].embedding

    def _cosine_similarity(self, a, b):
        return float(
            np.dot(a, b) /
            (np.linalg.norm(a) * np.linalg.norm(b))
        )
