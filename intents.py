"""Definições compartilhadas entre nlp.py e queries.py. Alterar só em acordo com o grupo."""

# Chaves de params; valor ausente = None. "ano" é inteiro.
PARAM_KEYS = ["categoria", "origem", "ano"]

# Valores válidos, escritos exatamente como no CSV.
CATEGORIES = ["vegetação", "solo exposto", "água"]
ORIGINS = ["satélite", "drone"]

UNKNOWN_INTENT = "unknown"

# Toda intenção aplica os params presentes na pergunta.
# Frases de referência (dataset de treinamento), sem pré-processamento.
INTENTS = {
    "count_images": {
        "required": [],
        "examples": [
            "quantas imagens estão catalogadas",
            "há quantas imagens",
            "qual o total de imagens cadastradas",
            "quantas imagens existem no catálogo",
            "qual a quantidade de imagens",
            "quantas fotos tem no banco",
            "quantas imagens de água existem",
            "quantas imagens foram feitas por drone",
            "quantas imagens de vegetação são de 2023",
            "quantas imagens são de vegetação",
            "quantas fotos são de vegetação",
            "contar as imagens de solo exposto",
        ],
    },
    "list_images": {
        "required": [],
        "examples": [
            "quais imagens possuem vegetação",
            "mostre as imagens de água",
            "exibir imagens de solo exposto",
            "exiba fotos de solo exposto",
            "mostre fotos de solo exposto",
            "mostre fotos feitas por satélite",
            "quero ver fotos de água",
            "mostrar imagens com drone",
            "exiba imagens por satélite",
            "listar imagens de vegetação feitas por satélite",
            "liste fotos de água",
            "quero ver as fotos de drone",
            "listar as imagens de satélite",
            "mostre todas as imagens",
            "quais imagens são de 2024",
            "ver imagens de vegetação feitas por drone",
            "exiba as fotos de água de satélite",
            "buscar imagens de solo exposto de 2015",
        ],
    },
    "count_by_category": {
        "required": [],
        "examples": [
            "quantas imagens há de cada categoria",
            "quantidade de imagens por categoria",
            "qual a distribuição das imagens por categoria",
            "quantas imagens tem cada categoria",
            "total de imagens por tipo",
            "quantas são de vegetação, água e solo exposto",
            "contagem por categoria",
            "resumo das imagens por categoria",
        ],
    },
    "count_by_origin": {
        "required": [],
        "examples": [
            "quantas imagens vieram de cada origem",
            "quantidade de imagens por origem",
            "quantas são de drone e quantas são de satélite",
            "qual a distribuição das imagens por origem",
            "total de imagens por fonte",
            "quantas imagens tem cada origem",
            "contagem por origem",
            "resumo das imagens por origem de captura",
        ],
    },
}
