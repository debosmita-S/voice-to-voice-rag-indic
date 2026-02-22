import os
import logging
from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

class RAGModule:
    """
    RAG module for contextual question answering.
    Uses FAISS for vector search and llama-cpp-python for local LLM inference.
    """
    def __init__(self, 
                 repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                 model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                 embedding_model="paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize RAG components and download model if necessary.
        """
        logging.info("Loading embedding model...")
        self.embedder = SentenceTransformer(embedding_model)
        
        logging.info(f"Checking for LLM model: {model_file}...")
        try:
            model_path = hf_hub_download(repo_id=repo_id, filename=model_file)
            logging.info(f"Model loaded from {model_path}")
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
            raise

        self.llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4)
        
        self.index = None
        self.documents = []

    def close(self):
        """
        Explicitly close the LLM and free resources.
        """
        if hasattr(self, 'llm') and self.llm is not None:
            logging.info("Closing LLM instance...")
            try:
                # Explicitly close to avoid issues during garbage collection
                self.llm.close()
            except Exception as e:
                logging.debug(f"Error while closing LLM: {e}")
            finally:
                self.llm = None

    def __del__(self):
        """
        Safer destructor to handle potential issues during garbage collection.
        """
        try:
            self.close()
        except:
            pass

    def index_documents(self, docs: List[str]):
        """
        Create a FAISS index from a list of strings.
        """
        self.documents = docs
        embeddings = self.embedder.encode(docs)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        logging.info(f"Indexed {len(docs)} document segments.")

    def retrieve(self, query: str, k=3):
        """
        Retrieve top-k relevant documents for a query.
        """
        if self.index is None:
            return []
            
        query_vec = self.embedder.encode([query])
        distances, indices = self.index.search(np.array(query_vec).astype('float32'), k)
        
        results = [self.documents[i] for i in indices[0] if i != -1]
        return results

    def generate_response(self, query: str, language="Hindi"):
        """
        Generate a response based on retrieved context.
        """
        context_docs = self.retrieve(query)
        context_text = "\n".join(context_docs)
        
        # TinyLlama Chat Template
        system_msg = f"You are a helpful assistant. Answer the question based on the context. Respond in {language}."
        user_msg = f"Context:\n{context_text}\n\nQuestion: {query}"
        
        prompt = f"<|system|>\n{system_msg}</s>\n<|user|>\n{user_msg}</s>\n<|assistant|>\n"
        
        logging.info("Generating LLM response...")
        # Stop on all common termination tokens to prevent loops or repetition
        output = self.llm(
            prompt, 
            max_tokens=150, 
            stop=["</s>", "<|user|>", "Question:", "Answer:"], 
            echo=False
        )
        
        response = output['choices'][0]['text'].strip()
        
        # Final cleanup for any leftover tokens (just in case)
        response = response.replace("<|assistant|>", "").replace("<|user|>", "").replace("<|system|>", "")
        return response

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # rag = RAGModule()
    # rag.index_documents(["भारत की राजधानी नई दिल्ली है।", "दिल्ली का लाल किला बहुत प्रसिद्ध है।"])
    # print(rag.generate_response("भारत की राजधानी क्या है?"))
