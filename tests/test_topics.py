"""Check document counting, outlier retention and saved embedding alignment."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load_script('07_analyze_topics')
fit = load_script('06_fit_topics')


class TopicChecks(unittest.TestCase):
    def setUp(self):
        self.cleaning = json.loads((ROOT / 'config/keyword_cleaning.json').read_text())
        self.assignments = pd.DataFrame({
            'dataset_id': ['001', '002', '003', '004'], 'topic_id': [0, 0, 1, -1],
            'analysis_text': ['reef reefs coral climate change', 'reef coral climate changer',
                              'corals ocean climate change', 'wage jobs worker'],
        })
        self.topics = pd.DataFrame({
            'Topic': [-1, 0, 1], 'Count': [1, 2, 1],
            'Name': ['outliers', 'reefs', 'oceans'],
            'top_keywords': ['jobs; wages', 'reef; coral; climate change', 'ocean; corals'],
        })
        self.embeddings = np.array([[1., 0., 0.], [.8, .2, 0.], [0., 1., 0.], [0., 0., 1.]])

    def diagnostic(self):
        return analysis.build_normalized_diagnostics('Positive', self.assignments, self.topics, 1, self.cleaning)

    def test_variant_union_and_phrase_boundaries(self):
        result = self.diagnostic()
        reef = result.loc[result.Topic.eq(0) & result['Canonical term'].eq('reef')].iloc[0]
        self.assertEqual(reef['Occurrences in Topic'], 3)
        self.assertEqual(reef['Topic articles containing canonical term'], 2)
        self.assertEqual(reef['Topic vs corpus lift'], 2)
        self.assertAlmostEqual(reef['Relevance'], .6 * np.log1p(3) + .4 * np.log(2))
        phrase = result.loc[result.Topic.eq(0) & result['Canonical term'].eq('climate change')].iloc[0]
        self.assertEqual(phrase['Topic articles containing canonical term'], 1)
        self.assertEqual(phrase['Corpus articles containing canonical term'], 2)

    def test_outliers_retained_and_network_counts_articles(self):
        result = self.diagnostic()
        self.assertIn(-1, result.Topic.unique())
        nodes, _, co, _ = analysis.term_network(self.assignments, result, 0, 1, self.cleaning, 100)
        self.assertEqual(int(nodes.loc[nodes.term.eq('reef'), 'articles'].iloc[0]), 2)
        self.assertEqual(co.loc['reef', 'coral'], 2)
        with tempfile.TemporaryDirectory() as td:
            points, _, labels, _ = analysis.centroid_tables('Positive', self.assignments, self.embeddings, Path(td))
            self.assertIn(-1, points.topic_id.tolist())
            self.assertNotIn('Positive Topic -1', labels)
            self.assertEqual(points.articles.sum(), 4)

    def test_embedding_alignment_and_order_signature(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            group = directory / 'positive'
            group.mkdir()
            self.assignments.to_csv(group / 'positive_article_topic_assignments.csv', index=False)
            self.topics.to_csv(group / 'positive_complete_topic_list.csv', index=False)
            np.save(group / 'positive_article_embeddings.npy', self.embeddings)
            signature = fit.document_signature(self.assignments.dataset_id.tolist(), self.assignments.analysis_text.tolist(), 'encoder')
            (group / 'positive_embedding_manifest.json').write_text(json.dumps({
                'embedding_model': 'encoder', 'document_sha256': signature,
            }))
            actual, _, _ = analysis.load_group(directory, 'Positive')
            self.assertEqual(actual.dataset_id.iloc[0], '001')
            self.assignments.iloc[::-1].to_csv(group / 'positive_article_topic_assignments.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'assignment order'):
                analysis.load_group(directory, 'Positive')
            self.assignments.to_csv(group / 'positive_article_topic_assignments.csv', index=False)
            np.save(group / 'positive_article_embeddings.npy', self.embeddings[:2])
            with self.assertRaisesRegex(ValueError, 'align'):
                analysis.load_group(directory, 'Positive')


if __name__ == '__main__':
    unittest.main()
