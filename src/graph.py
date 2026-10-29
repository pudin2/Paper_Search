"""Construcción y exportación del grafo interactivo."""
import textwrap
from html import escape
from pathlib import Path

import networkx as nx
from pyvis.network import Network

from .fetcher import Paper
from .relations import Relation

PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7"]
KEYWORD_COLOR = "#bab0ac"


def _paper_tooltip(p: Paper) -> str:
    abstract = textwrap.shorten(p.abstract, width=400, placeholder="…")
    return "\n".join([
        p.title,
        f"{', '.join(p.authors[:4])}{' et al.' if len(p.authors) > 4 else ''} · {p.published}",
        f"Categorías: {', '.join(p.categories)}",
        f"Keywords: {', '.join(p.keywords)}",
        "",
        textwrap.fill(abstract, width=80),
    ])


def _relation_tooltip(r: Relation) -> str:
    lines = [f"Similitud: {r.similarity:.2f}"]
    if r.shared_keywords:
        lines.append(f"Keywords: {', '.join(r.shared_keywords)}")
    if r.shared_categories:
        lines.append(f"Categorías: {', '.join(r.shared_categories)}")
    if r.shared_authors:
        lines.append(f"Autores: {', '.join(r.shared_authors)}")
    return "\n".join(lines)


def build_graph(
    papers: list[Paper],
    relations: list[Relation],
    concepts: dict[str, set[str]],
    show_keywords: bool = True,
) -> nx.Graph:
    """Grafo con nodos paper y, opcionalmente, nodos keyword compartidos (bipartito)."""
    G = nx.Graph()
    categories = list(dict.fromkeys(p.primary_category for p in papers))
    color_of = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(categories)}

    for p in papers:
        G.add_node(
            p.id,
            label=textwrap.shorten(p.title, width=40, placeholder="…"),
            title=_paper_tooltip(p),
            color=color_of[p.primary_category],
            shape="dot",
            size=25,
            group=p.primary_category,
        )

    for r in relations:
        G.add_edge(
            r.source,
            r.target,
            title=_relation_tooltip(r),
            weight=1 + 8 * max(r.similarity, 0),  # pyvis.from_nx traduce `weight` a grosor
            label=f"{r.similarity:.2f}",
            color="#666666",
            kind="paper",
        )

    if show_keywords:
        # Solo keywords que aparecen en 2+ papers: son las que explican las conexiones.
        counts: dict[str, list[str]] = {}
        for pid, kws in concepts.items():
            for kw in kws:
                counts.setdefault(kw, []).append(pid)
        for kw, pids in counts.items():
            if len(pids) < 2:
                continue
            node_id = f"kw::{kw}"
            G.add_node(node_id, label=kw, title=f"Keyword compartida por {len(pids)} papers",
                       color=KEYWORD_COLOR, shape="diamond", size=10)
            for pid in pids:
                G.add_edge(node_id, pid, color=KEYWORD_COLOR, weight=1, dashes=True, kind="keyword")

    return G


def export_html(G: nx.Graph, path: Path, heading: str) -> Path:
    # El `heading` de pyvis 0.3.2 se renderiza duplicado; lo insertamos a mano.
    net = Network(height="800px", width="100%", cdn_resources="remote", bgcolor="#ffffff")
    net.from_nx(G)
    net.barnes_hut(gravity=-8000, spring_length=200)
    html = net.generate_html().replace(
        "<body>",
        f'<body>\n<h2 style="font-family:sans-serif;text-align:center;margin:8px 0">{escape(heading)}</h2>',
        1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Escritura propia en UTF-8: write_html usa la codificación del sistema
    # y en Windows falla con nombres de autor con acentos.
    path.write_text(html, encoding="utf-8")
    return path
