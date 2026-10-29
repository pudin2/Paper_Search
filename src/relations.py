"""Cálculo de relaciones entre pares de artículos."""
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer

from .fetcher import Paper


@dataclass
class Relation:
    source: str
    target: str
    similarity: float
    shared_keywords: list[str]
    shared_categories: list[str]
    shared_authors: list[str]


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
    model: SentenceTransformer,
    concepts: dict[str, set[str]],
    sim_threshold: float = 0.4,
) -> list[Relation]:
    """Devuelve las relaciones relevantes entre cada par de papers.

    Un par se considera relacionado si la similitud semántica de sus abstracts
    supera `sim_threshold` o si comparten keywords o autores.
    """
    emb = model.encode([f"{p.title}. {p.abstract}" for p in papers], normalize_embeddings=True)
    sims = np.asarray(emb) @ np.asarray(emb).T

    relations = []
    for i, j in combinations(range(len(papers)), 2):
        a, b = papers[i], papers[j]
        rel = Relation(
            source=a.id,
            target=b.id,
            similarity=float(sims[i, j]),
            shared_keywords=sorted(concepts[a.id] & concepts[b.id]),
            shared_categories=sorted(set(a.categories) & set(b.categories)),
            shared_authors=sorted(set(a.authors) & set(b.authors)),
        )
        if rel.similarity >= sim_threshold or rel.shared_keywords or rel.shared_authors:
            relations.append(rel)
    return relations
