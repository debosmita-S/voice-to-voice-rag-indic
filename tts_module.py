import os
import subprocess
import logging
import shutil
from huggingface_hub import hf_hub_download

class TTSModule:
    """
    Text-to-Speech module using Piper.
    Piper is extremely fast and runs locally, making it ideal for Raspberry Pi.
    """
    def __init__(self, default_lang="hi"):
        """
        Initialize the TTS module.
        :param default_lang: Default language code ('hi' or 'en').
        """
        self.default_lang = default_lang
        # Language to model mapping
        self.voice_map = {
            "hi": {
                "model": "hi_IN-rohan-medium.onnx",
                "repo_path": "hi/hi_IN/rohan/medium/"
            },
            "en": {
                "model": "en_GB-alan-medium.onnx",
                "repo_path": "en/en_GB/alan/medium/"
            }
        }
        
        # Loaded models cache
        self.loaded_models = {}
        
        # 1. Try to find piper executable
        self.piper_path = shutil.which("piper")
        if not self.piper_path:
            # Fallback for common Windows pip installation path
            user_scripts = os.path.expandvars(r"%APPDATA%\Python\Python313\Scripts\piper.exe")
            if os.path.exists(user_scripts):
                self.piper_path = user_scripts
            else:
                self.piper_path = "piper"
        
        logging.info(f"Using Piper executable at: {self.piper_path}")

        # 2. Pre-load default language model
        self._load_model(self.default_lang)

    def _load_model(self, lang_code):
        """
        Download and load a model for a specific language.
        """
        if lang_code in self.loaded_models:
            return self.loaded_models[lang_code]
            
        # Default to English if language not supported
        if lang_code not in self.voice_map:
            logging.warning(f"Language {lang_code} not directly supported for TTS. Falling back to English.")
            lang_code = "en"
            
        config = self.voice_map[lang_code]
        model_file = config["model"]
        repo_path = config["repo_path"]
        
        try:
            logging.info(f"Downloading TTS model for '{lang_code}': {model_file}...")
            model_path = hf_hub_download(
                repo_id="rhasspy/piper-voices", 
                filename=f"{repo_path}{model_file}"
            )
            # Download config
            hf_hub_download(
                repo_id="rhasspy/piper-voices", 
                filename=f"{repo_path}{model_file}.json"
            )
            self.loaded_models[lang_code] = model_path
            return model_path
        except Exception as e:
            logging.error(f"Failed to load TTS model for {lang_code}: {e}")
            if lang_code != "en":
                return self._load_model("en")
            raise

    def speak(self, text, output_path="output.wav", lang_code=None):
        """
        Convert text to speech and save as a WAV file.
        :param text: Text to convert.
        :param output_path: Path to save the generated audio.
        :param lang_code: Language code for the voice. If None, uses default_lang.
        """
        if not text:
            return
            
        lang = lang_code if lang_code else self.default_lang
        model_path = self._load_model(lang)
        
        logging.info(f"Generating {lang} speech for: {text[:50]}...")
        
        # 1. Ensure the output file is deleted before generation
        # This prevents playing stale audio if the next command fails
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logging.debug(f"Deleted existing file: {output_path}")
            except Exception as e:
                logging.warning(f"Could not delete {output_path}: {e}")

        try:
            # 2. Use stdin for robust text input (avoids shell quoting/echo issues)
            # Command: piper --model model.onnx --output_file out.wav
            command = [
                self.piper_path,
                "--model", str(model_path),
                "--output_file", str(output_path)
            ]
            
            # Use subprocess with input argument to pipe text directly to Piper's stdin
            subprocess.run(command, input=text.encode('utf-8'), check=True, capture_output=True)
            
            if os.path.exists(output_path):
                logging.info(f"Speech successfully saved to {output_path}")
            else:
                logging.error(f"Piper finished but {output_path} was not created.")
                
        except subprocess.CalledProcessError as e:
            logging.error(f"Piper failed (code {e.returncode}): {e.stderr.decode()}")
        except Exception as e:
            logging.error(f"Unexpected error in TTS for language {lang}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # piper = TTSModule()
    # piper.speak("नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ?", "hindi_test.wav")
