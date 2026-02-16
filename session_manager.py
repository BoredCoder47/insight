import time


class SessionManager:
    def __init__(self, vision, verifier, audio):
        self.vision = vision
        self.verifier = verifier
        self.audio = audio
        self.last_identity_check = 0

    def process_frame(self, frame):
        vision_data = self.vision.process_frame(frame)

        current_time = time.time()
        identity_result = None

        if current_time - self.last_identity_check >= 5:
            identity_result = self.verifier.verify(frame)
            self.last_identity_check = current_time

        return {
            "vision": vision_data,
            "identity": identity_result
        }

    def process_audio(self, audio_array):
        return self.audio.process_audio_chunk(audio_array)
