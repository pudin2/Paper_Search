"""Busca artículos de arXiv sobre un topic y genera un grafo de sus relaciones.

Edita los parámetros de la sección CONFIGURACIÓN y ejecuta el archivo
(F5 en Spyder, o `python main.py`).
"""
import json
import re
import webbrowser
from dataclasses import asdict
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.fetcher import ArxivRateLimited, search_papers
from src.graph import build_graph, export_html
from src.keywords import extract_keywords
from src.relations import compute_relations, merge_keywords

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
TOPIC = "antifragility in oscillators"   # Tema a buscar en arXiv
NUM_PAPERS = 10                    # Número de artículos a analizar

SIM_THRESHOLD = 0.4               # Similitud mínima (0-1) para unir dos papers
KEYWORDS_PER_PAPER = 8            # Máximo de keywords extraídas por paper
KEYWORD_MERGE_THRESHOLD = 0.85    # Similitud para tratar dos keywords como la misma

SHOW_KEYWORDS = True              # Mostrar nodos de keywords compartidas en el grafo
USE_CACHE = True                  # False = volver a consultar arXiv aunque haya caché
OPEN_BROWSER = True               # Abrir el grafo en el navegador al terminar

MODEL_NAME = "all-MiniLM-L6-v2"   # Modelo de embeddings (sentence-transformers)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
# =============================================================================


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "topic"


def main() -> None:
    print(f"Buscando '{TOPIC}' en arXiv...")
    try:
        papers = search_papers(TOPIC, NUM_PAPERS, use_cache=USE_CACHE)
    except ArxivRateLimited as e:
        print(e)
        return
    if not papers:
        print("No se encontraron artículos.")
        return
    for i, p in enumerate(papers, 1):
        print(f"  {i}. [{p.id}] {p.title}")

    print(f"Cargando modelo {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print("Extrayendo keywords...")
    extract_keywords(papers, model, top_n=KEYWORDS_PER_PAPER)
    concepts = merge_keywords(papers, model, threshold=KEYWORD_MERGE_THRESHOLD)
    for p in papers:
        print(f"  [{p.id}] {', '.join(p.keywords)}")

    print("Calculando relaciones...")
    relations = compute_relations(papers, model, concepts, SIM_THRESHOLD)
    for r in sorted(relations, key=lambda r: -r.similarity):
        extra = f"  keywords: {', '.join(r.shared_keywords)}" if r.shared_keywords else ""
        print(f"  {r.source} <-> {r.target}  sim={r.similarity:.2f}{extra}")

    slug = slugify(TOPIC)
    G = build_graph(papers, relations, concepts, show_keywords=SHOW_KEYWORDS)
    html_path = export_html(G, OUTPUT_DIR / f"{slug}.html", heading=f"arXiv: {TOPIC}")

    json_path = OUTPUT_DIR / f"{slug}.json"
    json_path.write_text(
        json.dumps(
            {"topic": TOPIC, "papers": [asdict(p) for p in papers], "relations": [asdict(r) for r in relations]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nGrafo: {html_path}\nDatos: {json_path}")
    if OPEN_BROWSER:
        webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()
