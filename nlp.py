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

_SPELLING_ALIASES = {"qnts": "quantas"}


def _correct_word(word: str) -> str:
    """Corrige uma palavra quando há um único candidato próximo."""
    if word in _SPELLING_ALIASES:
        return _SPELLING_ALIASES[word]
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

# Calibrado com perguntas válidas e fora do domínio; test_nlp.py protege a acurácia.
_MIN_SIMILARITY = 0.7


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


def _unknown_value_after(
    tokens: list[str], source_tokens: list[str], markers: set[str]
) -> str | None:
    """Retorna um valor desconhecido que venha logo após uma preposição."""
    for index, token in enumerate(tokens[:-1]):
        if token not in markers:
            continue

        candidate = tokens[index + 1]
        if (
            not candidate.isalpha()
            or candidate in _SPELLING_VOCABULARY
            or index + 1 >= len(source_tokens)
        ):
            continue
        return source_tokens[index + 1]
    return None


def extract_params(text: str) -> dict:
    """Extrai filtros do catálogo ou preserva valores desconhecidos."""
    normalized_tokens = [
        _correct_word(word) for word in _normalize_text(text)
    ]
    source_tokens = (
        re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
        if isinstance(text, str)
        else []
    )
    params = {key: None for key in PARAM_KEYS}
    params["categoria"] = _first_value(normalized_tokens, CATEGORIES)
    if params["categoria"] is None:
        params["categoria"] = _unknown_value_after(
            normalized_tokens, source_tokens, {"de"}
        )

    params["origem"] = _first_value(normalized_tokens, ORIGINS)
    if params["origem"] is None:
        params["origem"] = _unknown_value_after(
            normalized_tokens, source_tokens, {"por"}
        )

    if isinstance(text, str):
        year_match = re.search(r"(?<!\d)\d{4}(?!\d)", text)
        if year_match:
            params["ano"] = int(year_match.group())

    return params


def classify_query(text: str) -> tuple[str, float]:
    """Classifica a pergunta e preserva consultas com filtros desconhecidos."""
    intent_id, score = _best_match(text)
    if intent_id != UNKNOWN_INTENT and score >= _MIN_SIMILARITY:
        return intent_id, score

    params = extract_params(text)
    has_unknown_filter = (
        params["categoria"] is not None
        and params["categoria"] not in CATEGORIES
    ) or (
        params["origem"] is not None
        and params["origem"] not in ORIGINS
    )
    tokens = set(_prepare_text(text).split())
    image_cues = {"imagem", "imagens", "foto", "fotos"}
    if has_unknown_filter and tokens.intersection(image_cues):
        count_cues = {
            "quantas", "quantos", "quantidade", "contar", "conte",
            "total", "numero",
        }
        if tokens.intersection(count_cues):
            return "count_images", score
        return "list_images", score

    return UNKNOWN_INTENT, score
