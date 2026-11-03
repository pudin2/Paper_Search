"""Texto completo de los papers para los pares dudosos.

Cuando la similitud de dos abstracts queda justo por debajo de θ, se descarga
el texto completo de ambos, se trocea y se comparan las dos nubes de fragmentos
con la similitud de Chamfer. Así se detectan papers que comparten una parte
concreta (un modelo, un método) aunque el tema general del abstract difiera.
"""
import io
import re
import time
from itertools import combinations
from pathlib import Path

import lxml.html
import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from .embedding import chunk_text, embed_chunks
from .fetcher import CACHE_DIR, Paper
from .relations import Relation

FULLTEXT_DIR = CACHE_DIR / "fulltext"
USER_AGENT = "Motor_Busqueda/0.1 (research prototype; few requests)"
REQUEST_DELAY = 3.0  # misma cortesía que pide arXiv para su API

# Bloques del HTML de arXiv (LaTeXML) que no son prosa: bibliografía (inflaría
# la similitud entre papers que citan lo mismo), ecuaciones, figuras, tablas...
_HTML_DROP = [
    "ltx_bibliography", "ltx_equation", "ltx_equationgroup", "ltx_figure", "ltx_table",
    "ltx_TOC", "ltx_note", "ltx_authors", "ltx_dates", "ltx_pagination",
]
_PDF_REFERENCES = re.compile(r"\n\s*(references|bibliography)\s*\n", re.IGNORECASE)

_last_request = 0.0


def _get(url: str) -> requests.Response:
    global _last_request
    wait = REQUEST_DELAY - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    try:
        return requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    finally:
        _last_request = time.monotonic()


def _from_html(paper_id: str) -> str | None:
    resp = _get(f"https://arxiv.org/html/{paper_id}")
    if resp.status_code != 200:
        return None
    root = lxml.html.fromstring(resp.content)
    docs = root.find_class("ltx_document")
    if not docs:
        return None
    doc = docs[0]
    for cls in _HTML_DROP:
        for el in doc.find_class(cls):
            el.drop_tree()
    # Matemática en línea: su texto es LaTeX ("\omega_{i}"), ruido para el modelo.
    for el in doc.iter("math"):
        el.drop_tree()
    paragraphs = [" ".join(p.text_content().split()) for p in doc.find_class("ltx_p")]
    text = "\n".join(p for p in paragraphs if p)
    return text or None


def _from_pdf(pdf_url: str | None) -> str | None:
    if not pdf_url:
        return None
    resp = _get(pdf_url)
    if resp.status_code != 200:
        return None
    reader = PdfReader(io.BytesIO(resp.content))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    matches = list(_PDF_REFERENCES.finditer(text))
    if matches:
        text = text[: matches[-1].start()]
    text = re.sub(r"-\n(?=[a-z])", "", text)  # palabras partidas por guion al final de línea
    return " ".join(text.split()) or None


def fetch_fulltext(paper: Paper) -> str | None:
    """Texto completo sin bibliografía ni ecuaciones (HTML de arXiv, o el PDF si no hay HTML).

    Se guarda en cache/fulltext/; un archivo vacío marca que no hubo texto disponible.
    """
    cache_file = FULLTEXT_DIR / f"{paper.id.replace('/', '_')}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8") or None

    text = None
    try:
        text = _from_html(paper.id) or _from_pdf(paper.pdf_url)
    except (requests.RequestException, ValueError) as e:
        print(f"    no se pudo descargar {paper.id}: {e}")
        return None  # error de red: no se cachea, se reintentará en la próxima ejecución

    FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(text or "", encoding="utf-8")
    return text


def chamfer(A: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray]:
    """Similitud de Chamfer entre dos nubes de puntos unitarios.

    ½ [ media_a max_b ⟨a,b⟩ + media_b max_a ⟨a,b⟩ ]: cada fragmento busca su
    fragmento más cercano en el otro paper. Devuelve también la matriz M = ABᵀ.
    """
    M = A @ B.T
    return 0.5 * (M.max(axis=1).mean() + M.max(axis=0).mean()), M


def _evidence(M: np.ndarray, chunks_a: list[str], chunks_b: list[str], k: int = 2) -> list[dict]:
    """Los k pares de pasajes más parecidos, sin repetir fragmento."""
    used_a, used_b, out = set(), set(), []
    for flat in np.argsort(M, axis=None)[::-1]:
        i, j = np.unravel_index(flat, M.shape)
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append({"a": chunks_a[i], "b": chunks_b[j], "similarity": float(M[i, j])})
        if len(out) == k:
            break
    return out


def add_fulltext_relations(
    papers: list[Paper],
    sims: np.ndarray,
    relations: list[Relation],
    model: SentenceTransformer,
    sim_threshold: float,
    band: float,
    fulltext_threshold: float,
) -> list[Relation]:
    """Revisa con texto completo los pares con θ − band ≤ s < θ.

    Si el par ya tenía arista (por keywords o autores), se le añade la similitud
    de texto completo y la evidencia. Si no la tenía y Chamfer ≥ `fulltext_threshold`,
    se crea una arista nueva. Devuelve la lista de relaciones actualizada.
    """
    pairs = [
        (i, j) for i, j in combinations(range(len(papers)), 2)
        if sim_threshold - band <= sims[i, j] < sim_threshold
    ]
    if not pairs:
        return relations

    by_pair = {frozenset((r.source, r.target)): r for r in relations}
    needed = sorted({k for pair in pairs for k in pair})
    print(f"  {len(pairs)} pares dudosos; texto completo de {len(needed)} papers...")

    clouds: dict[int, tuple[list[str], np.ndarray]] = {}
    for k in needed:
        text = fetch_fulltext(papers[k])
        if text:
            chunks = chunk_text(text, model)
            clouds[k] = (chunks, embed_chunks(chunks, model))

    result = list(relations)
    for i, j in pairs:
        if i not in clouds or j not in clouds:
            continue
        a, b = papers[i], papers[j]
        score, M = chamfer(clouds[i][1], clouds[j][1])
        existing = by_pair.get(frozenset((a.id, b.id)))
        if existing is None and score < fulltext_threshold:
            continue
        rel = existing or Relation(
            source=a.id,
            target=b.id,
            similarity=float(sims[i, j]),
            shared_keywords=[],
            shared_categories=sorted(set(a.categories) & set(b.categories)),
            shared_authors=[],
        )
        rel.fulltext_similarity = float(score)
        rel.evidence = _evidence(M, clouds[i][0], clouds[j][0])
        if existing is None:
            rel.via_fulltext = True
            result.append(rel)
    return result
