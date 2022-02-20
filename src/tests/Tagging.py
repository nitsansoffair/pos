import unittest
from collections import defaultdict

import numpy as np

from src.utilities.utils_pos import preprocess
from src.Tagging import Tagging


class TaggingTest(unittest.TestCase):
    def test_create_dictionaries(self):
        tagging = Tagging()
        target = tagging.create_dictionaries
        with open("../../data/WSJ_02-21.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/hmm_vocab.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        test_cases = [
            {
                "name": "default_case",
                "input": {
                    "training_corpus": training_corpus,
                    "vocab": vocab,
                    "verbose": False,
                },
                "expected": {
                    "len_emission_counts": 31140,
                    "len_transition_counts": 1421,
                    "len_tag_counts": 46,
                    "emission_counts": {
                        ("DT", "the"): 41098,
                        ("NNP", "--unk_upper--"): 4635,
                        ("NNS", "Arts"): 2,
                    },
                    "transition_counts": {
                        ("VBN", "TO"): 2142,
                        ("CC", "IN"): 1227,
                        ("VBN", "JJR"): 66,
                    },
                    "tag_counts": {"PRP": 17436, "UH": 97, ")": 1376, },
                },
            },
            {
                "name": "small_case",
                "input": {
                    "training_corpus": training_corpus[:1000],
                    "vocab": vocab,
                    "verbose": False,
                },
                "expected": {
                    "len_emission_counts": 442,
                    "len_transition_counts": 272,
                    "len_tag_counts": 38,
                    "emission_counts": {
                        ("DT", "the"): 48,
                        ("NNP", "--unk_upper--"): 9,
                        ("NNS", "Arts"): 1,
                    },
                    "transition_counts": {
                        ("VBN", "TO"): 3,
                        ("CC", "IN"): 2,
                        ("VBN", "JJR"): 1,
                    },
                    "tag_counts": {"PRP": 11, "UH": 0, ")": 2, },
                },
            },
        ]
        for test_case in test_cases:
            result_emission, result_transition, result_tag = target(**test_case["input"])
            self.assertEqual(True, isinstance(result_emission, defaultdict))
            self.assertEqual(True, len(result_emission) == test_case["expected"]["len_emission_counts"])
            for k, v in test_case["expected"]["emission_counts"].items():
                self.assertEqual(True, np.isclose(result_emission[k], v))
            self.assertEqual(True, isinstance(result_transition, defaultdict))
            self.assertEqual (True, len(result_transition) == test_case["expected"]["len_transition_counts"])
            for k, v in test_case["expected"]["transition_counts"].items():
                self.assertEqual(True, np.isclose(result_transition[k], v))
            self.assertEqual(True, isinstance(result_tag, defaultdict))
            self.assertEqual(True, len(result_tag) == test_case["expected"]["len_tag_counts"])
            for k, v in test_case["expected"]["tag_counts"].items():
                self.assertEqual(True, np.isclose(result_tag[k], v))

    def test_predict_pos(self):
        tagging = Tagging()
        target = tagging.predict_pos
        with open("../../data/WSJ_02-21.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/hmm_vocab.txt", 'r') as f:
            voc_l = f.read().split('\n')
        with open("../../data/WSJ_24.pos", 'r') as f:
            y = f.readlines()
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, prep = preprocess(vocab, "../../data/test.words")
        emission_counts, transition_counts, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
        states = sorted(tag_counts.keys())
        test_cases = [
            {
                "name": "default_check",
                "input": {
                    "prep": prep,
                    "y": y,
                    "emission_counts": emission_counts,
                    "vocab": vocab,
                    "states": states,
                },
                "expected": 0.8888563993099213,
            },
            {
                "name": "small_check",
                "input": {
                    "prep": prep[:1000],
                    "y": y[:1000],
                    "emission_counts": emission_counts,
                    "vocab": vocab,
                    "states": states,
                },
                "expected": 0.876,
            },
        ]
        for test_case in test_cases:
            result = target(**test_case["input"])
            self.assertEqual(True, np.isclose(result, test_case["expected"]))

    def test_create_transition_matrix(self):
        tagging = Tagging()
        target = tagging.create_transition_matrix
        with open("../../data/WSJ_02-21.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/hmm_vocab.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, transition_counts, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
        test_cases = [
            {
                "name": "default_check",
                "input": {
                    "alpha": 0.001,
                    "tag_counts": tag_counts,
                    "transition_counts": transition_counts,
                },
                "expected": {
                    "0:5": np.array(
                        [
                            [
                                7.03997297e-06,
                                7.03997297e-06,
                                7.03997297e-06,
                                7.03997297e-06,
                                7.03997297e-06,
                            ],
                            [
                                1.35647553e-07,
                                1.35647553e-07,
                                1.35647553e-07,
                                1.35647553e-07,
                                1.35647553e-07,
                            ],
                            [
                                1.44528595e-07,
                                1.44673124e-04,
                                6.93751711e-03,
                                6.79298851e-03,
                                5.05864537e-03,
                            ],
                            [
                                7.32039770e-07,
                                1.69101919e-01,
                                7.32039770e-07,
                                7.32039770e-07,
                                7.32039770e-07,
                            ],
                            [
                                7.26719892e-07,
                                7.27446612e-04,
                                7.26719892e-07,
                                7.27446612e-04,
                                7.26719892e-07,
                            ],
                        ]
                    ),
                    "30:35": np.array(
                        [
                            [
                                2.21706877e-06,
                                2.21706877e-06,
                                2.21706877e-06,
                                8.87049214e-03,
                                2.21706877e-06,
                            ],
                            [
                                3.75650909e-07,
                                7.51677469e-04,
                                3.75650909e-07,
                                5.10888993e-02,
                                3.75650909e-07,
                            ],
                            [
                                1.72277159e-05,
                                1.72277159e-05,
                                1.72277159e-05,
                                1.72277159e-05,
                                1.72277159e-05,
                            ],
                            [
                                4.47733569e-05,
                                4.47286283e-08,
                                4.47286283e-08,
                                8.95019852e-05,
                                4.47733569e-05,
                            ],
                            [
                                1.03043917e-05,
                                1.03043917e-05,
                                1.03043917e-05,
                                6.18366548e-02,
                                3.09234796e-02,
                            ],
                        ]
                    ),
                },
            },
            {
                "name": "alpha_check",
                "input": {
                    "alpha": 0.05,
                    "tag_counts": tag_counts,
                    "transition_counts": transition_counts,
                },
                "expected": {
                    "0:5": np.array(
                        [
                            [
                                3.46500347e-04,
                                3.46500347e-04,
                                3.46500347e-04,
                                3.46500347e-04,
                                3.46500347e-04,
                            ],
                            [
                                6.78030457e-06,
                                6.78030457e-06,
                                6.78030457e-06,
                                6.78030457e-06,
                                6.78030457e-06,
                            ],
                            [
                                7.22407640e-06,
                                1.51705604e-04,
                                6.94233742e-03,
                                6.79785589e-03,
                                5.06407756e-03,
                            ],
                            [
                                3.65416941e-05,
                                1.68859168e-01,
                                3.65416941e-05,
                                3.65416941e-05,
                                3.65416941e-05,
                            ],
                            [
                                3.62765726e-05,
                                7.61808024e-04,
                                3.62765726e-05,
                                7.61808024e-04,
                                3.62765726e-05,
                            ],
                        ]
                    ),
                    "30:35": np.array(
                        [
                            [
                                1.10302228e-04,
                                1.10302228e-04,
                                1.10302228e-04,
                                8.93448048e-03,
                                1.10302228e-04,
                            ],
                            [
                                1.87666554e-05,
                                7.69432872e-04,
                                1.87666554e-05,
                                5.10640694e-02,
                                1.87666554e-05,
                            ],
                            [
                                8.29187396e-04,
                                8.29187396e-04,
                                8.29187396e-04,
                                8.29187396e-04,
                                8.29187396e-04,
                            ],
                            [
                                4.69603252e-05,
                                2.23620596e-06,
                                2.23620596e-06,
                                9.16844445e-05,
                                4.69603252e-05,
                            ],
                            [
                                5.03524673e-04,
                                5.03524673e-04,
                                5.03524673e-04,
                                6.09264854e-02,
                                3.07150050e-02,
                            ],
                        ]
                    ),
                },
            },
        ]
        for test_case in test_cases:
            result = target(**test_case["input"])
            self.assertEqual(True, isinstance(result, np.ndarray))
            self.assertEqual(True, np.allclose(result[0:5, 0:5], test_case["expected"]["0:5"]))
            self.assertEqual(True, np.allclose(result[30:35, 30:35], test_case["expected"]["30:35"]))

if __name__ == '__main__':
    unittest.main()
