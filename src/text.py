"""Utilidades de normalización de texto."""
import re
import unicodedata


def normalize_name(text: str) -> str:
    """Forma canónica para comparar nombres: sin acentos, minúsculas, sin puntuación.

    "Oliver López-Corona" y "Oliver Lopez Corona" -> "oliver lopez corona".
    """
    decomposed = unicodedata.normalize("NFKD", text)
    no_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", no_accents.lower()).strip()


def slugify(text: str, max_len: int = 60) -> str:
    """Nombre de archivo legible: "Antifragility in Oscillators" -> "antifragility_in_oscillators"."""
    return normalize_name(text).replace(" ", "_")[:max_len].rstrip("_") or "topic"
