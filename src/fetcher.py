"""Búsqueda de artículos en arXiv."""
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import arxiv

# Relativa a la raíz del proyecto, no al directorio desde el que se ejecuta (Spyder puede usar otro).
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    url: str
    published: str
    keywords: list[str] = field(default_factory=list)


class ArxivRateLimited(RuntimeError):
    pass


def search_papers(topic: str, max_results: int = 5, use_cache: bool = True) -> list[Paper]:
    """Devuelve los `max_results` artículos más relevantes para `topic`.

    Los resultados se guardan en `cache/` para no repetir peticiones a arXiv,
    que limita a 1 petición cada 3 segundos y bloquea temporalmente (HTTP 429)
    a quien se pasa.
    """
    key = hashlib.sha1(f"{topic.lower().strip()}|{max_results}".encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{key}.json"
    if use_cache and cache_file.exists():
        return [Paper(**d) for d in json.loads(cache_file.read_text(encoding="utf-8"))]

    try:
        papers = _fetch(topic, max_results)
    except arxiv.HTTPError as e:
        if e.status in (429, 503):
            raise ArxivRateLimited(
                f"arXiv está limitando las peticiones (HTTP {e.status}). Espera unos minutos y vuelve a intentarlo."
            ) from e
        raise

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps([asdict(p) for p in papers], ensure_ascii=False, indent=2), encoding="utf-8")
    return papers


def _fetch(topic: str, max_results: int) -> list[Paper]:
    # Un solo reintento: reintentar contra un 429 solo alarga el bloqueo.
    client = arxiv.Client(page_size=max_results, delay_seconds=3, num_retries=1)
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return [
        Paper(
            id=r.get_short_id(),
            title=" ".join(r.title.split()),
            abstract=" ".join(r.summary.split()),
            authors=[a.name for a in r.authors],
            categories=list(r.categories),
            primary_category=r.primary_category,
            url=r.entry_id,
            published=r.published.date().isoformat(),
        )
        for r in client.results(search)
    ]
