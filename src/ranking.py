"""Reordenación semántica de los resultados de arXiv."""
import numpy as np
from sentence_transformers import SentenceTransformer

from .embedding import embed_documents
from .fetcher import Paper


def rerank(papers: list[Paper], model: SentenceTransformer, topic: str, top_n: int) -> list[Paper]:
    """Devuelve los `top_n` papers más cercanos al topic en el espacio de embeddings.

    El buscador de arXiv es léxico: para "antifragility in oscillators" devuelve
    cualquier paper que diga "antifragility", trate o no de osciladores. Pidiendo
    un conjunto amplio y ordenándolo por ⟨φ(q), φ(d)⟩ nos quedamos con los que
    tratan el tema completo. Rellena `paper.relevance` con esa similitud.
    """
    q = model.encode([topic], normalize_embeddings=True)
    docs = embed_documents([f"{p.title}. {p.abstract}" for p in papers], model)
    scores = docs @ np.asarray(q)[0]
    for paper, score in zip(papers, scores):
        paper.relevance = float(score)
    return sorted(papers, key=lambda p: -p.relevance)[:top_n]
