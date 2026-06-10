"""
HFModelManager — lazy-loading HuggingFace models for quality scoring and task classification.
Models: all-MiniLM-L6-v2 (task classification), BAAI/bge-large-en-v1.5 (session clustering).
"""

import logging
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

MODEL_REGISTRY = {
    "minilm":    "sentence-transformers/all-MiniLM-L6-v2",
    "bge_large": "BAAI/bge-large-en-v1.5",
}


class HFModelManager:
    _instance: Optional["HFModelManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._models: dict = {}
        self._timers: dict = {}
        self._lock = threading.Lock()
        self._device = self._detect_device()
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_instance(cls) -> "HFModelManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def _detect_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _get_model(self, key: str):
        with self._lock:
            if key in self._models:
                self._reset_idle_timer(key)
                return self._models[key]
        return self._load_model(key)

    def _load_model(self, key: str):
        model_id = MODEL_REGISTRY[key]
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(
                model_id,
                cache_folder=str(MODELS_DIR),
                device=self._device,
            )
            with self._lock:
                self._models[key] = model
            self._reset_idle_timer(key)
            logger.info(f"Loaded {key} ({model_id}) on {self._device}")
            return model
        except Exception as exc:
            logger.warning(f"Failed to load {key}: {exc}")
            return None

    def _reset_idle_timer(self, key: str):
        if key in self._timers:
            self._timers[key].cancel()
        timer = threading.Timer(600, self._unload_model, args=(key,))
        timer.daemon = True
        timer.start()
        self._timers[key] = timer

    def _unload_model(self, key: str):
        with self._lock:
            if key in self._models:
                del self._models[key]
                logger.info(f"Unloaded {key} after 600s idle")

    def encode(self, text: str) -> np.ndarray:
        model = self._get_model("minilm")
        if model is None:
            return self._tfidf_fallback_encode([text])[0]
        try:
            emb = model.encode(text, normalize_embeddings=True)
            return np.array(emb, dtype=np.float32)
        except Exception as exc:
            logger.debug(f"encode error: {exc}")
            return self._tfidf_fallback_encode([text])[0]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        model = self._get_model("minilm")
        if model is None:
            return self._tfidf_fallback_encode(texts)
        try:
            embs = model.encode(texts, normalize_embeddings=True, batch_size=32)
            return np.array(embs, dtype=np.float32)
        except Exception as exc:
            logger.debug(f"encode_batch error: {exc}")
            return self._tfidf_fallback_encode(texts)

    def encode_bge(self, text: str) -> np.ndarray:
        """BGE-large encoding for high-quality similarity tasks."""
        model = self._get_model("bge_large")
        if model is None:
            return self.encode(text)
        try:
            emb = model.encode(text, normalize_embeddings=True)
            return np.array(emb, dtype=np.float32)
        except Exception:
            return self.encode(text)

    def encode_bge_batch(self, texts: List[str]) -> np.ndarray:
        model = self._get_model("bge_large")
        if model is None:
            return self.encode_batch(texts)
        try:
            embs = model.encode(texts, normalize_embeddings=True, batch_size=16)
            return np.array(embs, dtype=np.float32)
        except Exception:
            return self.encode_batch(texts)

    def preload(self, keys: Optional[List[str]] = None):
        keys = keys or list(MODEL_REGISTRY.keys())
        for k in keys:
            if k in MODEL_REGISTRY:
                self._load_model(k)

    def _tfidf_fallback_encode(self, texts: List[str]) -> np.ndarray:
        """Deterministic fallback: character-level bigram frequency vector."""
        dim = 384
        result = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            text_lower = text.lower()
            for j in range(len(text_lower) - 1):
                bigram = text_lower[j:j+2]
                idx = (ord(bigram[0]) * 31 + ord(bigram[1])) % dim
                result[i, idx] += 1.0
            norm = np.linalg.norm(result[i])
            if norm > 0:
                result[i] /= norm
        return result
