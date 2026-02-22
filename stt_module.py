import os
from faster_whisper import WhisperModel
import logging

class STTModule:
    """
    Speech-to-Text module using faster-whisper.
    Supports multilingual transcription, optimized for CPU/GPU.
    """
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """
        Initialize the Whisper model.
        :param model_size: Size of the model (tiny, base, small, medium, large-v3)
        :param device: Device to run on (cpu, cuda)
        :param compute_type: Quantization type (int8, float16, etc.)
        """
        self.model_size = model_size
        logging.info(f"Loading Whisper model: {model_size} on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path, language=None):
        """
        Transcribe an audio file.
        :param audio_path: Path to the audio file.
        :param language: Target language code. If None, it will be auto-detected.
        :return: Tuple of (transcribed text, detected language code).
        """
        logging.info(f"Transcribing audio file: {audio_path}")
        # beam_size 5 is a good balance between speed and quality
        segments, info = self.model.transcribe(audio_path, beam_size=5, language=language)
        
        logging.info(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        
        full_text = []
        for segment in segments:
            logging.debug(f"[STT Segment] {segment.start:.2f}s -> {segment.end:.2f}s: {segment.text}")
            full_text.append(segment.text)
            
        return " ".join(full_text).strip(), info.language

if __name__ == "__main__":
    # Quick test if run directly
    logging.basicConfig(level=logging.INFO)
    stt = STTModule(model_size="tiny", device="cpu")
    # Note: Replace with a real audio file path for manual testing
    # print(stt.transcribe("test_audio.wav", language="hi"))
