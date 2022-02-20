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
        # todo: update tests after validation
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

    def test_create_emission_matrix(self):
        # todo: update tests after validation
        tagging = Tagging()
        target = tagging.create_emission_matrix
        with open("../../data/WSJ_02-21.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/hmm_vocab.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        emission_counts, transition_counts, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
        test_cases = [
            {
                "name": "default_check",
                "input": {
                    "alpha": 0.001,
                    "tag_counts": tag_counts,
                    "emission_counts": emission_counts,
                    "vocab": vocab,
                },
                "expected": {
                    "0:5": np.array(
                        [
                            [
                                6.03219988e-06,
                                6.03219988e-06,
                                8.56578416e-01,
                                6.03219988e-06,
                                6.03219988e-06,
                            ],
                            [
                                1.35212298e-07,
                                1.35212298e-07,
                                1.35212298e-07,
                                9.71365280e-01,
                                1.35212298e-07,
                            ],
                            [
                                1.44034584e-07,
                                1.44034584e-07,
                                1.44034584e-07,
                                1.44034584e-07,
                                1.44034584e-07,
                            ],
                            [
                                7.19539897e-07,
                                7.19539897e-07,
                                7.19539897e-07,
                                7.19539897e-07,
                                7.19539897e-07,
                            ],
                            [
                                7.14399508e-07,
                                7.14399508e-07,
                                7.14399508e-07,
                                7.14399508e-07,
                                7.14399508e-07,
                            ],
                        ]
                    ),
                    "30:35": np.array(
                        [
                            [
                                2.10625199e-06,
                                2.10625199e-06,
                                2.10625199e-06,
                                2.10625199e-06,
                                2.10625199e-06,
                            ],
                            [
                                3.72331731e-07,
                                3.72331731e-07,
                                3.72331731e-07,
                                3.72331731e-07,
                                3.72331731e-07,
                            ],
                            [
                                1.22283772e-05,
                                1.22406055e-02,
                                1.22283772e-05,
                                1.22283772e-05,
                                1.22283772e-05,
                            ],
                            [
                                4.46812012e-08,
                                4.46812012e-08,
                                4.46812012e-08,
                                4.46812012e-08,
                                4.46812012e-08,
                            ],
                            [
                                8.27972213e-06,
                                4.96866125e-02,
                                8.27972213e-06,
                                8.27972213e-06,
                                8.27972213e-06,
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
                    "emission_counts": emission_counts,
                    "vocab": vocab,
                },
                "expected": {
                    "0:5": np.array(
                        [
                            [
                                3.75699741e-05,
                                3.75699741e-05,
                                1.06736296e-01,
                                3.75699741e-05,
                                3.75699741e-05,
                            ],
                            [
                                5.84054154e-06,
                                5.84054154e-06,
                                5.84054154e-06,
                                8.39174848e-01,
                                5.84054154e-06,
                            ],
                            [
                                6.16686298e-06,
                                6.16686298e-06,
                                6.16686298e-06,
                                6.16686298e-06,
                                6.16686298e-06,
                            ],
                            [
                                1.95706206e-05,
                                1.95706206e-05,
                                1.95706206e-05,
                                1.95706206e-05,
                                1.95706206e-05,
                            ],
                            [
                                1.94943174e-05,
                                1.94943174e-05,
                                1.94943174e-05,
                                1.94943174e-05,
                                1.94943174e-05,
                            ],
                        ]
                    ),
                    "30:35": np.array(
                        [
                            [
                                3.04905937e-05,
                                3.04905937e-05,
                                3.04905937e-05,
                                3.04905937e-05,
                                3.04905937e-05,
                            ],
                            [
                                1.29841464e-05,
                                1.29841464e-05,
                                1.29841464e-05,
                                1.29841464e-05,
                                1.29841464e-05,
                            ],
                            [
                                4.01010547e-05,
                                8.42122148e-04,
                                4.01010547e-05,
                                4.01010547e-05,
                                4.01010547e-05,
                            ],
                            [
                                2.12351646e-06,
                                2.12351646e-06,
                                2.12351646e-06,
                                2.12351646e-06,
                                2.12351646e-06,
                            ],
                            [
                                3.88847844e-05,
                                4.70505891e-03,
                                3.88847844e-05,
                                3.88847844e-05,
                                3.88847844e-05,
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

    def test_initialize(self):
        # todo: update tests after validation
        tagging = Tagging()
        target = tagging.initialize
        with open("../../data/WSJ_02-21.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/hmm_vocab.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, corpus = preprocess(vocab, "../../data/test.words")
        emission_counts, transition_counts, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
        A = tagging.create_transition_matrix(0.001, tag_counts, transition_counts)
        B = tagging.create_emission_matrix(0.001, tag_counts, emission_counts, list(vocab))
        states = tag_counts.keys()
        test_cases = [
            {
                "name": "default_check",
                "input": {
                    "states": states,
                    "tag_counts": tag_counts,
                    "A": A,
                    "B": B,
                    "corpus": corpus,
                    "vocab": vocab,
                },
                "expected": {
                    "best_probs_shape": (46, 34199),
                    "best_paths_shape": (46, 34199),
                    "best_probs_col0": np.array(
                        [
                            -22.60982633,
                            -23.07660654,
                            -23.57298822,
                            -19.76726066,
                            -24.74325104,
                            -35.20241402,
                            -35.00096024,
                            -34.99203854,
                            -21.35069072,
                            -19.85767814,
                            -21.92098414,
                            -4.01623741,
                            -19.16380593,
                            -21.1062242,
                            -20.47163973,
                            -21.10157273,
                            -21.49584851,
                            -20.4811853,
                            -18.25856307,
                            -23.39717471,
                            -21.92146798,
                            -9.41377777,
                            -21.03053445,
                            -21.08029591,
                            -20.10863677,
                            -33.48185979,
                            -19.47301382,
                            -20.77150242,
                            -20.11727696,
                            -20.56031676,
                            -20.57193964,
                            -32.30366295,
                            -18.07551522,
                            -22.58887909,
                            -19.1585905,
                            -16.02994331,
                            -24.30968545,
                            -20.92932218,
                            -21.96797222,
                            -24.29571895,
                            -23.45968569,
                            -22.43665883,
                            -20.46568904,
                            -22.75551606,
                            -19.6637215,
                            -18.36288463,
                        ]
                    ),
                },
            }
        ]
        for test_case in test_cases:
            result_best_probs, result_best_paths = target(**test_case["input"])
            self.assertEqual(True, isinstance(result_best_probs, np.ndarray))
            self.assertEqual(True, isinstance(result_best_paths, np.ndarray))
            self.assertEqual(True, result_best_probs.shape == test_case["expected"]["best_probs_shape"])
            self.assertEqual(True, result_best_paths.shape == test_case["expected"]["best_paths_shape"])
            self.assertEqual(True, np.allclose(result_best_probs[:, 0], test_case["expected"]["best_probs_col0"]))
            self.assertEqual(True, np.all((result_best_paths == 0)))

if __name__ == '__main__':
    unittest.main()
