# Voice-to-Voice RAG System

A lightweight, local voice-to-voice interaction system using Retrieval-Augmented Generation (RAG). Optimized for low-resource environments and supporting multiple languages (Hindi & English).

## 🚀 Features
- **Dynamic STT**: Automatic language detection (Hindi/English) using `faster-whisper`.
- **Contextual RAG**: Intelligent response generation using `TinyLlama` and `FAISS` vector search.
- **Fast TTS**: Local speech synthesis using `Piper TTS`.
- **Dynamic VAD**: Voice Activity Detection for natural hands-free interaction.
- **Language Consistency**: Automatically responds in the same language detected from input.

## 🛠️ Tech Stack
- **Speech-to-Text**: `faster-whisper`
- **RAG/LLM**: `llama-cpp-python` (TinyLlama 1.1B Q4)
- **Vector DB**: `FAISS`
- **Embeddings**: `SentenceTransformers`
- **Text-to-Speech**: `Piper`

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd "RAG based V2V sys"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup models**:
   ```bash
   python setup.py
   ```

## 🎮 Usage
Run the main pipeline:
```bash
python main.py
```

## 📁 Project Structure
- `main.py`: Entry point and orchestration logic.
- `stt_module.py`: Speech-to-Text processing.
- `rag_module.py`: Document indexing and LLM response generation.
- `tts_module.py`: Text-to-Speech synthesis.
- `setup.py`: Utility to pre-download models.

## 📄 License
MIT
