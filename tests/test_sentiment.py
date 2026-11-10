"""Regression checks for input scope, full-token coverage, and article-ID joins."""
import importlib.util
from pathlib import Path
import unittest

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(filename):
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script("01_prepare_corpus.py")
vader = load_script("02_run_vader.py")
transformers_stage = load_script("03_run_transformers.py")
compare = load_script("04_compare_models.py")


class SentimentTests(unittest.TestCase):
    def test_prepare_retains_missing_body_but_model_excludes_it(self):
        source = pd.DataFrame({
            "dataset_id": ["01", "02", "03"],
            "title": ["Useful  tools!", "Missing article", "Risk report"],
            "body_text": ["Full\ntext  here.", " \t ", "Harms  discussed."],
            "quality_tier": ["background", "core", "high"],
            "summary": ["A summary that must not enter analysis.", "Publisher", "Summary"],
        })
        articles, _ = prepare.prepare_corpus(source)
        self.assertEqual(len(articles), 3)
        self.assertEqual(articles.loc[0, "analysis_text"], "Useful tools!\n\nFull text here.")
        self.assertEqual(articles["eligible_for_model"].tolist(), [True, False, True])
        self.assertEqual(vader.eligible_articles(articles)["dataset_id"].tolist(), ["01", "03"])
        self.assertFalse(articles["quality_tier_used_for_filtering"].any())

    def test_model_eligibility_cannot_include_missing_body(self):
        articles = pd.DataFrame({"dataset_id": ["x"], "body_text": [""],
                                 "analysis_text": ["Title only"], "eligible_for_model": [True]})
        with self.assertRaisesRegex(ValueError, "body-text availability"):
            vader.eligible_articles(articles)

    def test_chunk_weights_cover_all_tokens_at_boundaries(self):
        for length in (0, 1, 10, 11, 17, 18, 19, 100):
            with self.subTest(length=length):
                tokens = list(range(length))
                chunks = transformers_stage.article_chunks(tokens, content_window=10, stride=2)
                self.assertEqual(sum(weight for _, weight in chunks), length)
                self.assertEqual(sorted(set(value for chunk, _ in chunks for value in chunk)), tokens)
                for chunk, weight in chunks:
                    self.assertLessEqual(len(chunk), 10)
                    self.assertGreater(weight, 0)
                if length:
                    self.assertEqual(chunks[0][1], len(chunks[0][0]))
                    for chunk, weight in chunks[1:]:
                        self.assertEqual(weight, len(chunk) - 2)
        self.assertEqual(transformers_stage.article_chunks(list(range(11)), 10, 0),
                         [(list(range(10)), 10), ([10], 1)])

    def test_invalid_overlap_fails(self):
        for stride in (-1, 10, 11):
            with self.assertRaises(ValueError):
                transformers_stage.article_chunks([1], content_window=10, stride=stride)

    def test_label_mapping_uses_configuration(self):
        from types import SimpleNamespace
        model = SimpleNamespace(config=SimpleNamespace(id2label={0: "positive", 1: "negative", 2: "neutral"}))
        self.assertEqual(transformers_stage.label_mapping(model), {0: "Positive", 1: "Negative", 2: "Neutral"})
        model.config.id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
        with self.assertRaises(ValueError):
            transformers_stage.label_mapping(model)

    def make_frames(self):
        return [pd.DataFrame({"dataset_id": ["b", "a"], column: ["Negative", "Positive"]})
                for column in compare.MODEL_COLUMNS.values()]

    def test_compare_aligns_by_id_not_row_order(self):
        frames = self.make_frames()
        frames[1] = frames[1].iloc[::-1].reset_index(drop=True)
        joined, distribution, pairs, metrics = compare.compare_models(frames)
        self.assertEqual(metrics["all_three_agree_count"], 2)
        self.assertTrue(joined["vader_deberta_agree"].all())
        self.assertTrue(pairs["agreement_count"].eq(2).all())
        self.assertTrue(pairs["cohen_kappa"].eq(1).all())
        self.assertEqual(distribution["articles"].sum(), 6)

    def test_compare_rejects_mismatched_or_duplicate_ids(self):
        frames = self.make_frames()
        frames[1].loc[0, "dataset_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            compare.compare_models(frames)
        frames = self.make_frames()
        frames[2]["dataset_id"] = "a"
        with self.assertRaisesRegex(ValueError, "unique"):
            compare.compare_models(frames)

    def test_compare_rejects_unknown_labels_and_missing_model(self):
        frames = self.make_frames()
        frames[1].loc[0, "distilroberta_label"] = "Unknown"
        with self.assertRaisesRegex(ValueError, "Unexpected labels"):
            compare.compare_models(frames)
        with self.assertRaisesRegex(ValueError, "Exactly three"):
            compare.compare_models(frames[:2])

    def test_vader_standard_thresholds(self):
        self.assertEqual(vader.vader_label(0.05), "Positive")
        self.assertEqual(vader.vader_label(-0.05), "Negative")
        self.assertEqual(vader.vader_label(0.049999), "Neutral")
        self.assertEqual(vader.vader_label(-0.049999), "Neutral")


if __name__ == "__main__":
    unittest.main()
