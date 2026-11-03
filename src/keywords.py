"""Extracción de keywords a partir de los abstracts (arXiv no las provee)."""
import re

import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

from .fetcher import Paper
from .text import normalize_name

# Palabras que son sintagma nominal pero no concepto: una keyword formada solo
# por ellas ("paper", "novel approach", "analysis") se descarta.
GENERIC_WORDS = {
    "paper", "papers", "article", "work", "works", "study", "studies", "analysis",
    "approach", "approaches", "method", "methods", "result", "results", "property",
    "properties", "rate", "rates", "case", "cases", "problem", "problems", "way", "ways",
    "use", "number", "set", "type", "types", "term", "terms", "framework", "effect",
    "effects", "aspect", "aspects", "example", "examples", "question", "questions",
    "notion", "concept", "concepts", "idea", "ideas", "contribution", "contributions",
    "research", "literature", "review", "author", "authors", "field", "fields", "part",
    "order", "purpose", "goal", "goals", "fact", "context", "overview", "introduction",
    "conclusion", "new", "novel", "recent", "different", "various", "several", "present",
    "current", "high", "low", "large", "small", "important", "key", "main", "general",
    "specific", "based", "proposed", "existing", "related", "comparison", "minimum",
    "maximum", "internal", "parts", "following", "arbitrary",
}
# Restos de metadatos editoriales que a veces aparecen en el abstract
# ("© Springer Nature", "arXiv preprint"); basta una de estas palabras para descartar.
METADATA_WORDS = {
    "springer", "elsevier", "ieee", "acm", "wiley", "arxiv", "preprint", "copyright",
    "doi", "license", "licence", "publisher", "journal", "proceedings", "isbn", "creative",
    "commons", "manuscript", "authenticated", "version",
}

# Mismo patrón que usa el CountVectorizer interno de KeyBERT, para que los
# candidatos coincidan con su vocabulario (p. ej. "semi-supervised" -> "semi supervised").
_TOKEN = re.compile(r"(?u)\b\w\w+\b")
_SKIP_POS = {"DET", "PRON", "NUM", "PUNCT", "SYM"}

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _nlp


def _noun_phrases(text: str, max_words: int = 3) -> list[str]:
    """Frases nominales del texto ("graph neural networks", "node embeddings"...).

    Usarlas como candidatas evita que KeyBERT proponga combinaciones sin sentido
    como "networks present" o palabras sueltas como "useful".
    """
    phrases = set()
    for chunk in _get_nlp()(text).noun_chunks:
        words = [t.text for t in chunk if t.pos_ not in _SKIP_POS and not t.is_stop]
        tokens = _TOKEN.findall(" ".join(words).lower())
        if 1 <= len(tokens) <= max_words:
            phrases.add(" ".join(tokens))
    return sorted(phrases)


def _is_noise(phrase: str, author_tokens: list[set[str]]) -> bool:
    """True si la frase es genérica, un metadato editorial o el nombre de un autor."""
    tokens = normalize_name(phrase).split()
    if all(t in GENERIC_WORDS for t in tokens):
        return True
    if any(t in METADATA_WORDS for t in tokens):
        return True
    # "roman bauer" o "bauer" cuando Roman Bauer es autor del propio paper.
    return any(set(tokens) <= name for name in author_tokens)


def extract_keywords(papers: list[Paper], model: SentenceTransformer, top_n: int = 8) -> None:
    """Rellena `paper.keywords` con las frases nominales más representativas.

    MMR (Maximal Marginal Relevance) evita keywords casi repetidas dentro de un mismo paper.
    """
    kw_model = KeyBERT(model=model)
    for paper in papers:
        text = f"{paper.title}. {paper.abstract}"
        author_tokens = [set(normalize_name(a).split()) for a in paper.authors]
        candidates = [c for c in _noun_phrases(text) if not _is_noise(c, author_tokens)]
        if not candidates:
            continue
        results = kw_model.extract_keywords(
            text,
            candidates=candidates,
            keyphrase_ngram_range=(1, 3),
            use_mmr=True,
            diversity=0.5,
            top_n=top_n,
        )
        paper.keywords = [kw for kw, _ in results]
