"""Cálculo de relaciones entre pares de artículos."""
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer

from .embedding import embed_documents
from .fetcher import Paper
from .text import normalize_name


@dataclass
class Relation:
    source: str
    target: str
    similarity: float
    shared_keywords: list[str]
    shared_categories: list[str]
    shared_authors: list[str]
    # Rellenados por fulltext.py solo para pares dudosos (similitud de abstract cerca de θ).
    fulltext_similarity: float | None = None
    evidence: list[dict] = field(default_factory=list)  # pasajes más parecidos {"a", "b", "similarity"}
    via_fulltext: bool = False  # la arista existe solo gracias al texto completo


def similarity_matrix(papers: list[Paper], model: SentenceTransformer) -> np.ndarray:
    """S = ΦΦᵀ con Φ los embeddings (troceados si hace falta) de título + abstract."""
    emb = embed_documents([f"{p.title}. {p.abstract}" for p in papers], model)
    return emb @ emb.T


def merge_keywords(papers: list[Paper], model: SentenceTransformer, threshold: float = 0.85) -> dict[str, set[str]]:
    """Agrupa keywords semánticamente equivalentes entre papers.

    KeyBERT puede devolver "neural networks" en un paper y "neural network" en
    otro; sin agruparlas no aparecerían como compartidas. Con all-MiniLM-L6-v2,
    las variantes de un mismo término dan ~0.89-0.94 y conceptos vecinos pero
    distintos ("deep" vs "convolutional neural networks") ~0.76, de ahí el 0.85.
    Devuelve, por id de paper, el conjunto de conceptos canónicos que contiene.
    """
    unique = list(dict.fromkeys(kw for p in papers for kw in p.keywords))
    if not unique:
        return {p.id: set() for p in papers}

    emb = model.encode(unique, normalize_embeddings=True)
    canonical: dict[str, str] = {}
    centers: list[int] = []
    for i, kw in enumerate(unique):
        match = next((c for c in centers if float(emb[i] @ emb[c]) >= threshold), None)
        if match is None:
            centers.append(i)
            canonical[kw] = kw
        else:
            canonical[kw] = unique[match]

    return {p.id: {canonical[kw] for kw in p.keywords} for p in papers}


def compute_relations(
    papers: list[Paper],
    sims: np.ndarray,
    concepts: dict[str, set[str]],
    sim_threshold: float = 0.4,
) -> list[Relation]:
    """Devuelve las relaciones relevantes entre cada par de papers.

    Un par se considera relacionado si la similitud semántica de sus abstracts
    (`sims`, de `similarity_matrix`) supera `sim_threshold` o si comparten
    keywords o autores.
    """
    relations = []
    for i, j in combinations(range(len(papers)), 2):
        a, b = papers[i], papers[j]
        rel = Relation(
            source=a.id,
            target=b.id,
            similarity=float(sims[i, j]),
            shared_keywords=sorted(concepts[a.id] & concepts[b.id]),
            shared_categories=sorted(set(a.categories) & set(b.categories)),
            shared_authors=_shared_authors(a.authors, b.authors),
        )
        if rel.similarity >= sim_threshold or rel.shared_keywords or rel.shared_authors:
            relations.append(rel)
    return relations


def _shared_authors(authors_a: list[str], authors_b: list[str]) -> list[str]:
    """Autores comunes, comparando nombres normalizados.

    arXiv no unifica la grafía de un mismo autor entre papers: "López-Corona"
    y "Lopez Corona" son la misma persona. Se devuelve la grafía del primer paper.
    """
    keys_a = {normalize_name(n): n for n in authors_a}
    keys_b = {normalize_name(n) for n in authors_b}
    return sorted(keys_a[k] for k in keys_a.keys() & keys_b)
