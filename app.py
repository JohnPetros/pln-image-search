"""Servidor Flask: página única e endpoint de busca."""

from flask import Flask, jsonify, render_template, request

from nlp import classify_query, extract_params
from queries import run_intent

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/search")
def search():
    """Executa a pipeline de PLN sobre ?q= e devolve o resultado em JSON."""
    text = request.args.get("q", "").strip()
    if not text:
        return jsonify({
            "query": text,
            "intent": None,
            "score": 0.0,
            "params": {},
            "answer": "Digite uma pergunta para começar.",
            "images": [],
        })

    intent_id, score = classify_query(text)
    params = extract_params(text)
    result = run_intent(intent_id, params)

    return jsonify({
        "query": text,
        "intent": intent_id,
        "score": round(float(score), 4),
        "params": params,
        "answer": result["answer"],
        "images": result["images"],
    })


if __name__ == "__main__":
    app.run(debug=True)
