import numpy as np


class AudioMonitor:
    def __init__(self, threshold=0.02):
        self.threshold = threshold

    def process_audio_chunk(self, audio_array):
        rms = np.sqrt(np.mean(np.square(audio_array)))
        speaking = rms > self.threshold
        return {
            "rms": float(rms),
            "speaking_detected": speaking
        }
