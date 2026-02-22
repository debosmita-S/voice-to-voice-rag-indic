import os
import logging
from stt_module import STTModule
from rag_module import RAGModule
from tts_module import TTSModule

def setup_system():
    logging.basicConfig(level=logging.INFO)
    print("=== Voice-to-Voice System Setup ===")
    
    try:
        print("\n1. Initializing STT (Whisper Base)...")
        stt = STTModule(model_size="base")
        
        print("\n2. Initializing RAG (TinyLlama)...")
        rag = RAGModule()
        
        print("\n3. Initializing TTS (Hindi & English)...")
        tts = TTSModule()
        tts._load_model("en")
        
        print("\n=== Setup Complete! ===")
        print("You can now run 'python main.py' to start the system.")
        print("Note: The system uses 'hi_IN-rohan-medium.onnx' by default as 'uma' was not found.")
        print("Note: To test without a microphone, place a .wav file as 'input.wav' and modify main.py.")
        
    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")
        print("Please check your internet connection and ensure all dependencies in requirements.txt are installed.")

if __name__ == "__main__":
    setup_system()
