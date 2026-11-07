<p align="center">
  <img src="static/logo.png" alt="Logo do projeto" width="160">
</p>

# PC.6 — Busca semântica de imagens de sensoriamento remoto

Trabalho prático PC.6 da disciplina de Processamento de Linguagem Natural (Prof. Fabrício Galende Marques de Carvalho).

Motor de busca em linguagem natural sobre um catálogo de imagens de sensoriamento remoto. O usuário digita uma pergunta, o sistema a representa como **Bag of Words**, compara com frases de referência por **similaridade de cosseno** e, de acordo com a frase de referência mais similar, executa a consulta correspondente em um **DataFrame Pandas** carregado de um arquivo CSV.

**Stack:** Python 3, Flask, Pandas, scikit-learn e NLTK.

## Como executar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Depois, abra http://127.0.0.1:5000 no navegador.

## Arquitetura

![Arquitetura do sistema](docs/arquitetura.png)

O diagrama editável está em [`docs/arquitetura.svg`](docs/arquitetura.svg).

## Fluxo de uma busca

```
pergunta do usuário
  → classify_query(text)            → (intent_id, score)
  → extract_params(text)            → params
  → run_intent(intent_id, params)   → resposta + imagens
  → exibição na página
```

Exemplo:

| Pergunta | Intenção | Consulta Pandas | Resposta |
|---|---|---|---|
| "Quantas imagens estão catalogadas?" | `count_all` | `len(df)` | "No total, há X imagens cadastradas." |
| "Quais imagens possuem vegetação?" | `list_by_category` | `df[df["categoria"] == "vegetação"]` | Lista de imagens exibidas na página |

## Estrutura e responsáveis

| Arquivo | Responsável | Conteúdo |
|---|---|---|
| `intents.py` | Todos | Intenções, frases de referência, chaves de `params` e valores válidos. Alterar só em acordo com o grupo. |
| `data/metadata.csv` | Pessoa 1 | Catálogo das imagens. |
| `static/imagens/` | Pessoa 1 | Arquivos das imagens (reais). |
| `queries.py` | Pessoa 1 | `run_intent`: uma consulta Pandas por intenção. |
| `nlp.py` | Pessoa 2 | `classify_query` e `extract_params`: pré-processamento, correção de digitação, BOW e cosseno. |
| `app.py` | Pessoa 3 | Servidor Flask com as rotas `/` (página) e `/search` (JSON). |
| `templates/index.html`, `static/script.js`, `static/style.css` | Pessoa 3 | Página única: campo de busca, resposta e galeria de imagens. |
| `requirements.txt`, `README.md` | Pessoa 3 | Dependências e documentação. |
| `docs/arquitetura.svg`, `docs/arquitetura.png` | Pessoa 3 | Diagrama da arquitetura (editar o SVG e regenerar o PNG). |
| `static/logo.png` | — | Logo do projeto. |

## Base de metadados

Arquivo `data/metadata.csv`, uma linha por imagem:

| Coluna | Conteúdo | Exemplo |
|---|---|---|
| `arquivo` | Caminho relativo à raiz do projeto | `static/imagens/img_001.jpg` |
| `categoria` | `vegetação`, `solo exposto` ou `água` | `água` |
| `data` | Data da imagem no formato `dd/mm/yyyy` | `15/03/2024` |
| `origem` | Origem da imagem | `satélite`, `drone` |

## Contrato entre os módulos

### `classify_query(text) -> (intent_id, score)` — `nlp.py`

- O vetorizador BOW é ajustado (`fit`) **somente** nas frases de referência de `intents.py`. Na pergunta do usuário usa-se `transform`, nunca `fit_transform`.
- A pergunta é comparada por similaridade de cosseno com cada frase de referência. A intenção é a da **frase de referência mais similar**, como pede o enunciado.
- `score` é o **cosseno dessa frase mais similar**.
- Se o score ficar abaixo do limiar, ou se o vetor da pergunta for todo zero, retorna `"unknown"`. O limiar deve ser calibrado com perguntas de teste.

### `extract_params(text) -> dict` — `nlp.py`

- Retorna um dicionário com todas as chaves de `params` definidas em `intents.py`, por exemplo `{"categoria": "água", "origem": "drone", "ano": None}`.
- As chaves usam os mesmos nomes das colunas do CSV.
- Parâmetro não mencionado na pergunta vem como `None`.

### `run_intent(intent_id, params) -> dict` — `queries.py`

- Retorna `{"answer": str, "images": [...]}`. Cada imagem traz as colunas do CSV.
- `"unknown"` responde "Não entendi a pergunta. Tente reformular."
- Se faltar um parâmetro obrigatório da intenção, responde perguntando qual valor o usuário deseja (por exemplo, "Qual categoria? Opções: vegetação, solo exposto, água.").

## Convenções

- Código (nomes de variáveis, funções e comentários) em **inglês**.
- Dados, colunas do CSV, interface e documentação em **português**.

## Requisitos do enunciado

- Aplicação de página única com Python, Flask e Pandas.
- Metadados armazenados em arquivo `.csv`; consultas feitas sobre o DataFrame correspondente.
- Pipeline de PLN estudada em aula, com representação Bag of Words e similaridade de cosseno.
- Robustez a erros de digitação.
- Entregáveis: apresentação, código-fonte, procedimento de execução e `requirements.txt`.

**Prazo de entrega:** 16/11/2026.
