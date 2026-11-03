"""Embeddings de textos de cualquier longitud.

all-MiniLM-L6-v2 solo ve 256 tokens y trunca el resto sin avisar (algunos
abstracts ya lo superan). Los textos largos se trocean en fragmentos que caben
y cada documento pasa a ser una nube de puntos en la esfera.
"""
import re

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_TOKENS = 200  # holgura bajo el límite de 256 (incluye tokens especiales)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def chunk_text(text: str, model: SentenceTransformer, max_tokens: int = CHUNK_TOKENS) -> list[str]:
    """Agrupa frases consecutivas en fragmentos de como mucho `max_tokens` tokens.

    Se corta por frases, no por tokens, para que cada fragmento sea legible:
    se muestran como evidencia en el grafo.
    """
    def n_tokens(s: str) -> int:
        return len(model.tokenizer(s, add_special_tokens=False)["input_ids"])

    chunks, current, current_len = [], [], 0
    for sentence in _SENTENCE_END.split(text.strip()):
        length = n_tokens(sentence)
        if length > max_tokens:
            # Frase demasiado larga (p. ej. texto extraído de un PDF sin puntos): cortar por palabras.
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            words, piece = sentence.split(), []
            for w in words:
                if piece and n_tokens(" ".join(piece + [w])) > max_tokens:
                    chunks.append(" ".join(piece))
                    piece = []
                piece.append(w)
            if piece:
                chunks.append(" ".join(piece))
            continue
        if current and current_len + length > max_tokens:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += length
    if current:
        chunks.append(" ".join(current))
    return chunks


def embed_chunks(chunks: list[str], model: SentenceTransformer) -> np.ndarray:
    """Matriz (m × 384) con un vector unitario por fragmento."""
    return np.asarray(model.encode(chunks, normalize_embeddings=True))


def embed_documents(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Un vector unitario por documento: el centroide normalizado de sus fragmentos.

    Para un texto que cabe en un fragmento coincide con `model.encode`.
    """
    vectors = []
    for text in texts:
        centroid = embed_chunks(chunk_text(text, model), model).mean(axis=0)
        vectors.append(centroid / np.linalg.norm(centroid))
    return np.vstack(vectors)
