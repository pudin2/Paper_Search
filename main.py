"""Busca artículos de arXiv sobre un topic y genera un grafo de sus relaciones.

Edita los parámetros de la sección CONFIGURACIÓN y ejecuta el archivo
(F5 en Spyder, o `python main.py`).
"""
import json
import webbrowser
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.fetcher import ArxivRateLimited, search_papers
from src.graph import build_graph, export_html
from src.keywords import extract_keywords
from src.ranking import rerank
from src.fulltext import add_fulltext_relations
from src.relations import compute_relations, merge_keywords, similarity_matrix
from src.text import slugify

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
TOPIC = "antifragility in oscillators"   # Tema a buscar en arXiv
NUM_PAPERS = 10                    # Número de artículos a analizar
CANDIDATE_POOL = 50               # Artículos que se piden a arXiv antes de quedarse con
                                  # los NUM_PAPERS más cercanos al tema por significado

SIM_THRESHOLD = 0.4               # Similitud mínima (0-1) para unir dos papers
KEYWORDS_PER_PAPER = 8            # Máximo de keywords extraídas por paper
KEYWORD_MERGE_THRESHOLD = 0.85    # Similitud para tratar dos keywords como la misma

USE_FULLTEXT = True               # Revisar con el texto completo los pares que quedan cerca de θ
FULLTEXT_BAND = 0.10              # Se revisan los pares con SIM_THRESHOLD - BAND <= s < SIM_THRESHOLD
FULLTEXT_THRESHOLD = 0.40         # Chamfer mínimo para crear la arista. Calibrado con
                                  # "antifragility in oscillators": equivale a θ = 0.4 en abstracts

SHOW_KEYWORDS = True              # Mostrar nodos de keywords compartidas en el grafo
USE_CACHE = True                  # False = volver a consultar arXiv aunque haya caché
OPEN_BROWSER = True               # Abrir el grafo en el navegador al terminar

MODEL_NAME = "all-MiniLM-L6-v2"   # Modelo de embeddings (sentence-transformers)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
# =============================================================================


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    started = datetime.now().astimezone()
    print(f"Buscando '{TOPIC}' en arXiv...")
    try:
        candidates = search_papers(TOPIC, max(CANDIDATE_POOL, NUM_PAPERS), use_cache=USE_CACHE)
    except ArxivRateLimited as e:
        print(e)
        return
    if not candidates:
        print("No se encontraron artículos.")
        return

    print(f"Cargando modelo {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Reordenando {len(candidates)} candidatos por similitud con el tema...")
    papers = rerank(candidates, model, TOPIC, NUM_PAPERS)
    for i, p in enumerate(papers, 1):
        print(f"  {i}. ({p.relevance:.2f}) [{p.id}] {p.title}")

    print("Extrayendo keywords...")
    extract_keywords(papers, model, top_n=KEYWORDS_PER_PAPER)
    concepts = merge_keywords(papers, model, threshold=KEYWORD_MERGE_THRESHOLD)
    for p in papers:
        print(f"  [{p.id}] {', '.join(p.keywords)}")

    print("Calculando relaciones...")
    sims = similarity_matrix(papers, model)
    relations = compute_relations(papers, sims, concepts, SIM_THRESHOLD)
    if USE_FULLTEXT:
        relations = add_fulltext_relations(
            papers, sims, relations, model, SIM_THRESHOLD, FULLTEXT_BAND, FULLTEXT_THRESHOLD
        )
    for r in sorted(relations, key=lambda r: -r.similarity):
        extra = f"  keywords: {', '.join(r.shared_keywords)}" if r.shared_keywords else ""
        if r.fulltext_similarity is not None:
            extra += f"  texto completo={r.fulltext_similarity:.2f}" + (" (nueva)" if r.via_fulltext else "")
        print(f"  {r.source} <-> {r.target}  sim={r.similarity:.2f}{extra}")

    # Una carpeta por ejecución (tema + fecha): cambiar un umbral y volver a correr
    # no pisa el resultado anterior, y run.json dice con qué parámetros se obtuvo.
    run_dir = OUTPUT_DIR / f"{slugify(TOPIC)}_{started:%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    G = build_graph(papers, relations, concepts, show_keywords=SHOW_KEYWORDS)
    html_path = export_html(G, run_dir / "grafo.html", heading=f"arXiv: {TOPIC}")

    run = {
        "topic": TOPIC,
        "date": started.isoformat(timespec="seconds"),
        "source": "arXiv",
        "candidates_fetched": len(candidates),
        "parameters": {
            "num_papers": NUM_PAPERS,
            "candidate_pool": CANDIDATE_POOL,
            "sim_threshold": SIM_THRESHOLD,
            "keywords_per_paper": KEYWORDS_PER_PAPER,
            "keyword_merge_threshold": KEYWORD_MERGE_THRESHOLD,
            "use_fulltext": USE_FULLTEXT,
            "fulltext_band": FULLTEXT_BAND,
            "fulltext_threshold": FULLTEXT_THRESHOLD,
            "model": MODEL_NAME,
        },
    }
    write_json(run_dir / "run.json", run)
    write_json(run_dir / "papers.json", [asdict(p) for p in papers])
    write_json(run_dir / "relations.json", [asdict(r) for r in relations])

    print(f"\nResultados en: {run_dir}\n  grafo.html, run.json, papers.json, relations.json")
    if OPEN_BROWSER:
        webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()
