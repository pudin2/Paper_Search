"""Extracción de keywords a partir de los abstracts (arXiv no las provee)."""
import re

import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

from .fetcher import Paper

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


def extract_keywords(papers: list[Paper], model: SentenceTransformer, top_n: int = 8) -> None:
    """Rellena `paper.keywords` con las frases nominales más representativas.

    MMR (Maximal Marginal Relevance) evita keywords casi repetidas dentro de un mismo paper.
    """
    kw_model = KeyBERT(model=model)
    for paper in papers:
        text = f"{paper.title}. {paper.abstract}"
        candidates = _noun_phrases(text)
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
