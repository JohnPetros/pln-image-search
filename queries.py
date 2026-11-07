"""Consultas Pandas sobre o catálogo de imagens."""

from pathlib import Path

import pandas as pd

from intents import CATEGORIES, INTENTS, ORIGINS, UNKNOWN_INTENT

CSV_PATH = Path(__file__).parent / "data" / "metadata.csv"
IMAGE_COLUMNS = ["arquivo", "categoria", "data", "origem"]
VALID_VALUES = {"categoria": CATEGORIES, "origem": ORIGINS}


def load_metadata(path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True, dtype=str)
    # Converte a data para permitir filtro por ano.
    df["data_dt"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    return df


def apply_filters(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Aplica os params que não são None."""
    if params.get("categoria"):
        df = df[df["categoria"] == params["categoria"]]
    if params.get("origem"):
        df = df[df["origem"] == params["origem"]]
    if params.get("ano"):
        df = df[df["data_dt"].dt.year == int(params["ano"])]
    return df


def describe_filters(params: dict) -> str:
    """Ex.: ' de água, capturadas por drone, de 2024'."""
    parts = []
    if params.get("categoria"):
        parts.append(f"de {params['categoria']}")
    if params.get("origem"):
        parts.append(f"capturadas por {params['origem']}")
    if params.get("ano"):
        parts.append(f"de {int(params['ano'])}")
    return (" " + ", ".join(parts)) if parts else ""


def plural(n: int) -> str:
    return f"{n} imagem" if n == 1 else f"{n} imagens"


def to_images(df: pd.DataFrame) -> list[dict]:
    """Mais recentes primeiro, só com as colunas do CSV."""
    return df.sort_values("data_dt", ascending=False)[IMAGE_COLUMNS].to_dict("records")


# Uma função por intenção: (df, params) -> resultado.

def count_images(df: pd.DataFrame, params: dict) -> dict:
    filtered = apply_filters(df, params)
    description = describe_filters(params)
    if not description:
        return {"answer": f"No total, há {plural(len(filtered))} cadastradas.", "images": []}
    return {"answer": f"Há {plural(len(filtered))}{description}.", "images": []}


def list_images(df: pd.DataFrame, params: dict) -> dict:
    filtered = apply_filters(df, params)
    description = describe_filters(params)
    if filtered.empty:
        return {"answer": f"Nenhuma imagem encontrada{description}.", "images": []}
    return {"answer": f"Encontrei {plural(len(filtered))}{description}.",
            "images": to_images(filtered)}


def count_by_column(df: pd.DataFrame, params: dict, column: str, label: str) -> dict:
    filtered = apply_filters(df, params)
    counts = filtered[column].value_counts()
    if counts.empty:
        return {"answer": f"Nenhuma imagem encontrada{describe_filters(params)}.", "images": []}
    summary = "; ".join(f"{value}: {plural(n)}" for value, n in counts.items())
    return {"answer": f"Imagens{describe_filters(params)}, por {label} — {summary}.",
            "images": []}


def count_by_category(df: pd.DataFrame, params: dict) -> dict:
    # Ignora o filtro de categoria, pois ela é o agrupamento.
    return count_by_column(df, {**params, "categoria": None}, "categoria", "categoria")


def count_by_origin(df: pd.DataFrame, params: dict) -> dict:
    # Ignora o filtro de origem, pois ela é o agrupamento.
    return count_by_column(df, {**params, "origem": None}, "origem", "origem")


HANDLERS = {
    "count_images": count_images,
    "list_images": list_images,
    "count_by_category": count_by_category,
    "count_by_origin": count_by_origin,
}


def invalid_param_answer(params: dict) -> str | None:
    """Mensagem para valor inexistente no catálogo; None se todos forem válidos."""
    for key, valid in VALID_VALUES.items():
        value = params.get(key)
        if value is not None and value not in valid:
            return f"Não conheço {key} '{value}'. Opções: {', '.join(valid)}."
    return None


def missing_param_answer(param: str) -> str:
    valid = VALID_VALUES.get(param)
    if valid:
        return f"Qual {param}? Opções: {', '.join(valid)}."
    return f"Informe {param} na pergunta."


def run_intent(intent_id: str, params: dict) -> dict:
    """Executa a consulta da intenção e retorna {"answer", "images"}."""
    if intent_id == UNKNOWN_INTENT or intent_id not in HANDLERS:
        return {"answer": "Não entendi a pergunta. Tente reformular.", "images": []}

    invalid = invalid_param_answer(params)
    if invalid:
        return {"answer": invalid, "images": []}

    for param in INTENTS[intent_id]["required"]:
        if params.get(param) is None:
            return {"answer": missing_param_answer(param), "images": []}

    return HANDLERS[intent_id](load_metadata(), params)
