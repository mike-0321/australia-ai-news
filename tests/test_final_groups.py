"""Identity and missing-decision checks at the sentiment/topic boundary."""
import importlib.util
from pathlib import Path
import unittest

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/05_build_final_groups.py"
spec = importlib.util.spec_from_file_location("final_groups", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FinalGroupsTests(unittest.TestCase):
    def setUp(self):
        self.articles = pd.DataFrame({"dataset_id": ["a", "b", "c", "empty"],
                                      "body_text": ["benefits", "harm", "facts", " "],
                                      "analysis_text": ["A benefits", "B harm", "C facts", "Empty"]})
        self.vader = pd.DataFrame({"dataset_id": ["c", "a", "b"],
                                   "vader_label": ["Positive", "Positive", "Negative"]})
        self.deberta = pd.DataFrame({"dataset_id": ["b", "c", "a"],
                                     "deberta_label": ["Negative", "Neutral", "Positive"]})
        self.decisions = pd.DataFrame({"dataset_id": ["c"], "final_sentiment_label": ["Neutral"]})

    def test_labels_join_by_id_and_missing_body_is_excluded(self):
        result, summary = module.build_groups(self.articles, self.vader, self.deberta, self.decisions)
        self.assertEqual(result.set_index("dataset_id")["final_sentiment_label"].to_dict(),
                         {"a": "Positive", "b": "Negative", "c": "Neutral"})
        self.assertEqual(summary["excluded_missing_body"], 1)
        self.assertEqual(summary["vader_deberta_agreements"], 2)

    def test_unresolved_disagreement_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing decisions"):
            module.build_groups(self.articles, self.vader, self.deberta, self.decisions.iloc[:0])

    def test_unmatched_prediction_fails(self):
        with self.assertRaisesRegex(ValueError, "exactly match"):
            module.build_groups(self.articles, self.vader.iloc[:2], self.deberta, self.decisions)

    def test_agreement_cannot_be_overwritten(self):
        decisions = pd.concat([self.decisions, pd.DataFrame({"dataset_id": ["a"], "final_sentiment_label": ["Negative"]})])
        with self.assertRaisesRegex(ValueError, "contradicts"):
            module.build_groups(self.articles, self.vader, self.deberta, decisions)

    def test_duplicate_decisions_fail(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            module.build_groups(self.articles, self.vader, self.deberta, pd.concat([self.decisions, self.decisions]))

    def test_unknown_decision_id_fails(self):
        decisions = pd.DataFrame({"dataset_id": ["elsewhere"], "final_sentiment_label": ["Neutral"]})
        with self.assertRaisesRegex(ValueError, "outside"):
            module.build_groups(self.articles, self.vader, self.deberta, decisions)


if __name__ == "__main__":
    unittest.main()
