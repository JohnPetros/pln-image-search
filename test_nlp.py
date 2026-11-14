"""Testes da classificação de perguntas e da pipeline Flask."""

import unittest

import nlp
from app import app
from intents import INTENTS, PARAM_KEYS, UNKNOWN_INTENT


CALIBRATION_CASES = [
    ("quantas fotos existem no banco", "count_images"),
    ("Mostre as fotos de solo exposto de 2015", "list_images"),
    ("mostre fotos de satélite", "list_images"),
    ("quero ver imagens de água", "list_images"),
    ("mostrar imagens de drone", "list_images"),
    ("exiba imagens de água por satélite", "list_images"),
    ("quantas fotos de vegetação existem", "count_images"),
    ("qnts imagens de vegetacao", "count_images"),
    ("resumo rápido por categoria", "count_by_category"),
    ("resumo rápido por origem", "count_by_origin"),
    ("por", UNKNOWN_INTENT),
    ("qual é a capital do Brasil?", UNKNOWN_INTENT),
    ("quantas pessoas moram no brasil", UNKNOWN_INTENT),
    ("quantos gols o brasil fez", UNKNOWN_INTENT),
    ("geografia de outro planeta", UNKNOWN_INTENT),
]


MIN_CALIBRATION_ACCURACY = 1.0


class ClassifyQueryTests(unittest.TestCase):
    def test_reference_examples_keep_their_intent(self):
        for intent_id, definition in INTENTS.items():
            for example in definition["examples"]:
                with self.subTest(intent=intent_id, example=example):
                    classified, score = nlp.classify_query(example)
                    self.assertEqual(classified, intent_id)
                    self.assertAlmostEqual(score, 1.0)

    def test_unseen_questions_cover_all_intents(self):
        questions = {
            "quantas fotos existem no banco!": "count_images",
            "exiba imagens de água por satélite": "list_images",
            "resumo rápido por categoria": "count_by_category",
            "resumo rápido por origem": "count_by_origin",
        }
        for question, expected in questions.items():
            with self.subTest(question=question):
                intent_id, score = nlp.classify_query(question)
                self.assertEqual(intent_id, expected)
                self.assertGreaterEqual(score, nlp._MIN_SIMILARITY)

    def test_typo_correction_and_accents(self):
        question = "LISTAR imagens com vejetacao feitas por satelte!"
        intent_id, score = nlp.classify_query(question)
        self.assertEqual(intent_id, "list_images")
        self.assertGreaterEqual(score, nlp._MIN_SIMILARITY)

    def test_unknown_and_empty_queries(self):
        for question in ("", "!!!", "zxqv", "qual é a capital do Brasil?"):
            with self.subTest(question=question):
                intent_id, score = nlp.classify_query(question)
                self.assertEqual(intent_id, UNKNOWN_INTENT)
                self.assertGreaterEqual(score, 0.0)

    def test_unknown_result_preserves_nonzero_similarity(self):
        best_intent, best_score = nlp._best_match("por")
        intent_id, score = nlp.classify_query("por")
        self.assertNotEqual(best_intent, UNKNOWN_INTENT)
        self.assertEqual(intent_id, UNKNOWN_INTENT)
        self.assertEqual(score, best_score)
        self.assertGreater(score, 0.0)

    def test_similarity_threshold_meets_minimum_accuracy(self):
        correct = sum(
            nlp.classify_query(question)[0] == expected
            for question, expected in CALIBRATION_CASES
        )
        accuracy = correct / len(CALIBRATION_CASES)
        self.assertGreaterEqual(accuracy, MIN_CALIBRATION_ACCURACY)

    def test_unknown_filter_queries_keep_an_image_intent(self):
        self.assertEqual(nlp.classify_query("imagens de floresta")[0], "list_images")
        self.assertEqual(
            nlp.classify_query("quantas imagens de floresta")[0],
            "count_images",
        )


class ExtractParamsTests(unittest.TestCase):
    def test_extracts_accented_values_and_first_year(self):
        params = nlp.extract_params(
            "Imagens de SOLO EXPOSTO, de satélite em 2024 e drone em 2022"
        )
        self.assertEqual(
            params,
            {"categoria": "solo exposto", "origem": "satélite", "ano": 2024},
        )

    def test_extracts_typo_corrected_values(self):
        params = nlp.extract_params("imagens de vejetacao feitas por satelte")
        self.assertEqual(params["categoria"], "vegetação")
        self.assertEqual(params["origem"], "satélite")

    def test_extracts_unknown_category_and_origin(self):
        category_params = nlp.extract_params("imagens de floresta")
        self.assertEqual(
            category_params,
            {"categoria": "floresta", "origem": None, "ano": None},
        )

        origin_params = nlp.extract_params("imagens feitas por balão")
        self.assertEqual(
            origin_params,
            {"categoria": None, "origem": "balão", "ano": None},
        )

    def test_known_vocabulary_and_year_are_not_unknown_values(self):
        params = nlp.extract_params("imagens de cada categoria feitas por drone em 2024")
        self.assertEqual(
            params,
            {"categoria": None, "origem": "drone", "ano": 2024},
        )

    def test_correction_handles_transpositions_and_word_lengths(self):
        self.assertEqual(nlp._correct_word("sateliet"), "satelite")
        self.assertEqual(nlp._correct_word("vegetacau"), "vegetacao")
        self.assertEqual(nlp._correct_word("agu"), "agu")

    def test_missing_params_are_none(self):
        params = nlp.extract_params("mostre todas as imagens")
        self.assertEqual(list(params), PARAM_KEYS)
        self.assertEqual(params, {key: None for key in PARAM_KEYS})

    def test_value_matching_respects_word_boundaries(self):
        params = nlp.extract_params("imagem de aguamarinha e droneiro em 12024")
        self.assertEqual(params, {key: None for key in PARAM_KEYS})

    def test_abbreviated_count_query_uses_count_intent(self):
        intent_id, _ = nlp.classify_query("qnts imagens de vegetacao")
        self.assertEqual(intent_id, "count_images")

    def test_preserves_important_tokens(self):
        prepared = nlp._prepare_text("quantas quais imagens por cada categoria")
        self.assertEqual(prepared, "quantas quais imagens por cada categoria")


class SearchIntegrationTests(unittest.TestCase):
    def test_search_returns_filtered_images(self):
        client = app.test_client()
        response = client.get(
            "/search",
            query_string={
                "q": "listar imagens de vegetação feitas por drone em 2016"
            },
        )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["intent"], "list_images")
        self.assertEqual(
            result["params"],
            {"categoria": "vegetação", "origem": "drone", "ano": 2016},
        )
        self.assertEqual(len(result["images"]), 1)
        self.assertEqual(result["images"][0]["data"], "24/04/2016")

    def test_unknown_category_returns_catalog_options(self):
        response = app.test_client().get(
            "/search", query_string={"q": "imagens de floresta"}
        )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["intent"], "list_images")
        self.assertEqual(result["params"]["categoria"], "floresta")
        self.assertEqual(
            result["answer"],
            "Não conheço categoria 'floresta'. Opções: vegetação, solo exposto, água.",
        )


if __name__ == "__main__":
    unittest.main()
