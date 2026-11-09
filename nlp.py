"""Pipeline de PLN para classificar perguntas e extrair filtros."""

import re
import unicodedata

from nltk.metrics.distance import edit_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from intents import CATEGORIES, INTENTS, ORIGINS, PARAM_KEYS, UNKNOWN_INTENT


# Artigos e preposições não ajudam a distinguir as intenções.
_STOPWORDS = {
    "a", "as", "ao", "aos", "da", "das", "de", "do", "dos",
    "e", "em", "na", "nas", "no", "nos", "o", "os", "qual",
    "que", "um", "uma", "uns", "umas",
}


def _normalize_text(text: str) -> list[str]:
    """Remove acentos e pontuação e separa as palavras."""
    if not isinstance(text, str):
        return []

    without_accents = "".join(
        char for char in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(char)
    )
    return re.findall(r"[a-z0-9]+", without_accents)


def _build_spelling_vocabulary() -> tuple[str, ...]:
    words = set()
    texts = [
        example
        for definition in INTENTS.values()
        for example in definition["examples"]
    ]
    texts.extend(CATEGORIES)
    texts.extend(ORIGINS)
    for text in texts:
        words.update(_normalize_text(text))
    return tuple(sorted(words))


_SPELLING_VOCABULARY = _build_spelling_vocabulary()


def _correct_word(word: str) -> str:
    """Corrige uma palavra quando há um único candidato próximo."""
    if len(word) < 4 or not word.isalpha() or word in _SPELLING_VOCABULARY:
        return word

    max_distance = 1 if len(word) <= 7 else 2
    candidates = []
    for candidate in _SPELLING_VOCABULARY:
        if abs(len(candidate) - len(word)) > max_distance:
            continue
        distance = edit_distance(word, candidate, transpositions=True)
        if distance <= max_distance:
            candidates.append((distance, candidate))

    if not candidates:
        return word

    best_distance = min(distance for distance, _ in candidates)
    best_candidates = [
        candidate for distance, candidate in candidates
        if distance == best_distance
    ]
    return best_candidates[0] if len(best_candidates) == 1 else word


def _prepare_text(text: str) -> str:
    """Normaliza, corrige erros próximos e remove palavras funcionais."""
    tokens = (_correct_word(word) for word in _normalize_text(text))
    return " ".join(word for word in tokens if word not in _STOPWORDS)


_REFERENCE_EXAMPLES = [
    (intent_id, example)
    for intent_id, definition in INTENTS.items()
    for example in definition["examples"]
]
_REFERENCE_INTENTS = [intent_id for intent_id, _ in _REFERENCE_EXAMPLES]

_VECTORIZER = CountVectorizer(
    tokenizer=str.split,
    token_pattern=None,
    preprocessor=_prepare_text,
    lowercase=False,
)
_REFERENCE_MATRIX = _VECTORIZER.fit_transform(
    [example for _, example in _REFERENCE_EXAMPLES]
)

# Calibrado com perguntas variadas; a validação em test_nlp.py protege este valor.
_MIN_SIMILARITY = 0.65


def _best_match(text: str) -> tuple[str, float]:
    """Retorna a intenção e o cosseno da referência mais similar."""
    prepared = _prepare_text(text)
    if not prepared:
        return UNKNOWN_INTENT, 0.0

    query_vector = _VECTORIZER.transform([prepared])
    if query_vector.nnz == 0:
        return UNKNOWN_INTENT, 0.0

    similarities = cosine_similarity(query_vector, _REFERENCE_MATRIX)[0]
    best_index = int(similarities.argmax())
    return _REFERENCE_INTENTS[best_index], float(similarities[best_index])


def classify_query(text: str) -> tuple[str, float]:
    """Classifica a pergunta pela frase de referência mais similar."""
    intent_id, score = _best_match(text)
    if intent_id == UNKNOWN_INTENT or score < _MIN_SIMILARITY:
        return UNKNOWN_INTENT, score
    return intent_id, score


def _first_value(tokens: list[str], values: list[str]) -> str | None:
    """Encontra o primeiro valor válido, respeitando limites entre palavras."""
    matches = []
    for value in values:
        value_tokens = _normalize_text(value)
        size = len(value_tokens)
        for start in range(len(tokens) - size + 1):
            if tokens[start:start + size] == value_tokens:
                matches.append((start, -size, value))
    if not matches:
        return None
    return min(matches)[2]


def extract_params(text: str) -> dict:
    """Extrai os filtros mencionados e mantém os valores do catálogo."""
    normalized_tokens = [
        _correct_word(word) for word in _normalize_text(text)
    ]
    params = {key: None for key in PARAM_KEYS}
    params["categoria"] = _first_value(normalized_tokens, CATEGORIES)
    params["origem"] = _first_value(normalized_tokens, ORIGINS)

    if isinstance(text, str):
        year_match = re.search(r"(?<!\d)\d{4}(?!\d)", text)
        if year_match:
            params["ano"] = int(year_match.group())

    return params
