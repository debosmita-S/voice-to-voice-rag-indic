import os
import logging
import time
import winsound
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from stt_module import STTModule
from rag_module import RAGModule
from tts_module import TTSModule

class VoiceToVoiceSystem:
    def __init__(self, language="hi"):
        self.language = language
        # Upgrade to 'base' for better accuracy (user recommendation)
        self.stt = STTModule(model_size="base")
        self.rag = RAGModule()
        self.tts = TTSModule()
        
        # Load sample knowledge base (in production, this would be from files/database)
        self.rag.index_documents([
            "यह एक हल्का RAG-आधारित वॉयस-टू-वॉयस सिस्टम है।",
            "यह रास्पबेरी पाई जैसे कम संसाधन वाले उपकरणों पर चलने के लिए डिज़ाइन किया गया है।",
            "सिस्टम स्पीच-टू-टेक्स्ट के लिए व्हिस्पर और टेक्स्ट-टू-स्पीच के लिए पाइपर का उपयोग करता है।"
        ])

    def run_interaction(self, audio_input_path):
        """
        One full cycle: Hear -> Think -> Speak
        """
        print("\n--- Processing Interaction ---")
        
        # 1. Listen (STT)
        start_time = time.time()
        print(f"[DEBUG] Processing audio: {audio_input_path}")
        # Now returns text and language code (e.g., 'hi')
        user_text, lang_code = self.stt.transcribe(audio_input_path, language=None)
        
        # Verification step
        print(f"\n>>> Verified Transcription [{lang_code}]: \"{user_text}\"")
        
        if not user_text:
            print("[WARNING] No speech detected in transcription.")
            return
        
        # 2. Think (RAG)
        # Map code to full name for the prompt
        lang_map = {"hi": "Hindi", "en": "English"}
        target_lang = lang_map.get(lang_code, "English")
        
        response_text = self.rag.generate_response(user_text, language=target_lang)
        print(f"RAG Output [{target_lang}]: {response_text}")
        
        # 3. Speak (TTS)
        self.tts.speak(response_text, "response.wav", lang_code=lang_code)
        print(f"TTS completed in {time.time() - start_time:.2f} seconds")
        print(f"Response saved to: response.wav")
        
        # 4. Play (Audio Playback)
        print("Playing response...")
        play_audio("response.wav")
        
        print("--- Cycle Finished ---")

    def close(self):
        """
        Cleanup resources.
        """
        logging.info("Shutting down system...")
        if hasattr(self, 'rag'):
            self.rag.close()

def record_audio(filename, fs=16000, threshold=0.01, silence_duration=1.5, min_duration=2.0, timeout=15):
    """
    Records audio dynamically: starts on speech, stops on silence.
    Has a minimum duration to avoid cutting off early speech.
    """
    chunk_size = 1024
    silence_chunks_limit = int(silence_duration * fs / chunk_size)
    min_chunks = int(min_duration * fs / chunk_size)
    
    print("\n[READY] Listening for your voice... (Speak now)")
    print(f"[DEBUG] VAD Settings: Threshold={threshold}, SilenceDur={silence_duration}s, MinDur={min_duration}s")
    
    audio_data = []
    is_recording = False
    silence_chunks = 0
    start_time = time.time()
    
    try:
        with sd.InputStream(samplerate=fs, channels=1, blocksize=chunk_size, dtype='float32') as stream:
            while True:
                chunk, overflowed = stream.read(chunk_size)
                
                # Calculate energy (RMS)
                volume_norm = np.linalg.norm(chunk) / np.sqrt(len(chunk))
                
                # Periodically show level for debugging
                if int(time.time() * 10) % 5 == 0:
                    print(f"Level: {volume_norm:.4f}", end="\r")
                
                if not is_recording:
                    # Waiting for speech to start
                    if volume_norm > threshold:
                        is_recording = True
                        audio_data.append(chunk)
                        print("\n>>> Recording started...")
                    elif time.time() - start_time > timeout:
                        print("\nListening timed out (no speech detected).")
                        return False
                else:
                    # Currently recording
                    audio_data.append(chunk)
                    
                    if volume_norm < threshold:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0
                    
                    # Only stop if we've passed the minimum duration AND hit the silence limit
                    if len(audio_data) > min_chunks:
                        if silence_chunks > silence_chunks_limit:
                            print("\nDone! (Silence detected)")
                            break
                    elif silence_chunks > silence_chunks_limit * 2:
                        # Extra safety: even if below min duration, if it's EXTREMELY quiet for too long
                        # we assume it was a false start
                        if len(audio_data) < min_chunks / 2:
                             print("\nFalse start detected. Resetting...")
                             is_recording = False
                             audio_data = []
                             silence_chunks = 0
        
        if audio_data:
            # Reconstruct full recording
            recording = np.concatenate(audio_data, axis=0)
            
            # Simple Peak Normalization
            max_val = np.max(np.abs(recording))
            if max_val > 0:
                recording = recording / max_val
            
            # Convert to 16-bit PCM for Whisper/WAV
            recording = (recording * 32767).astype(np.int16)
            write(filename, fs, recording)
            print(f"Successfully recorded {len(recording)/fs:.2f} seconds.")
            return True
    except Exception as e:
        print(f"\nRecording error: {e}")
    
    return False

def play_audio(filename):
    """
    Utility to play a WAV file on Windows.
    """
    try:
        # winsound.PlaySound handles absolute paths and spaces gracefully
        winsound.PlaySound(filename, winsound.SND_FILENAME)
    except Exception as e:
        print(f"Playback failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    print("\n--- Initializing Voice-to-Voice System ---")
    system = None
    try:
        system = VoiceToVoiceSystem()
        print("Initialization Complete.\n")

        while True:
            print("\nOptions:")
            print("1. Record audio (5 seconds)")
            print("2. Provide audio file path")
            print("3. Quit")
            
            choice = input("Select an option (1/2/3): ").strip()
            
            if choice == '1':
                input_file = "input.wav"
                if record_audio(input_file):
                    system.run_interaction(input_file)
            elif choice == '2':
                input_file = input("Enter path to .wav file: ").strip()
                if os.path.exists(input_file):
                    system.run_interaction(input_file)
                else:
                    print(f"Error: File '{input_file}' not found.")
            elif choice == '3':
                print("Exiting...")
                break
            else:
                print("Invalid option. Please try again.")

    except Exception as e:
        print(f"Critical System Error: {e}")
    finally:
        if system:
            system.close()
