import pickle
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
            self.assertEqual(True, len(result_transition) == test_case["expected"]["len_transition_counts"])
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
        with open("../../data/WSJ_24.pos", 'r') as f:
            testing_corpus = f.readlines()
        with open("../../data/hmm_vocab.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, transition_counts, tag_counts = tagging.create_dictionaries(testing_corpus, vocab)
        test_cases = [
            {
                "name": "default_check",
                "input": {
                    "alpha": 0.001,
                    "tag_counts": tag_counts,
                    "transition_counts": transition_counts,
                },
                "expected": {
                    "0:5": [[3.50484551e-04, 4.50973479e-01, 3.50134417e-07, 7.00618968e-04, 3.50134417e-07],
                            [7.04705480e-03, 1.25081534e-01, 1.95992289e-02, 1.87183746e-02, 2.42256960e-03],
                            [3.34396715e-06, 5.08286351e-01, 3.34396715e-06, 3.34396715e-06, 3.34396715e-06],
                            [5.90185403e-03, 5.90185403e-03, 2.94945229e-06, 2.94945229e-06, 7.90456162e-01],
                            [2.09892421e-01, 6.03942791e-02, 9.90053918e-07, 9.90053918e-07, 7.92142140e-03]],
                    "30:35": np.array(
                        [[1.85027569e-05, 1.85027569e-05, 1.85027569e-05, 1.85027569e-05,
                          1.85027569e-05],
                         [1.11054350e-05, 1.11054350e-05, 1.11165404e-02, 1.11054350e-05,
                          1.11054350e-05],
                         [1.07473723e-05, 1.07473723e-05, 1.07473723e-05, 1.07473723e-05,
                          1.07473723e-05],
                         [8.12704192e-06, 8.12704192e-06, 8.13516896e-03, 8.12704192e-06,
                          8.12704192e-06],
                         [1.14881787e-05, 1.14881787e-05, 1.14996668e-02, 1.14881787e-05,
                          1.14881787e-05]]
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
                    "0:5":
                        [[3.67351223e-04, 4.50634993e-01, 1.74929154e-05, 7.17209530e-04,
                          1.74929154e-05],
                         [7.05434376e-03, 1.25030264e-01, 1.96002905e-02, 1.87198732e-02,
                          2.43215284e-03],
                         [1.65947561e-04, 5.04646532e-01, 1.65947561e-04, 1.65947561e-04,
                          1.65947561e-04],
                         [6.00644594e-03,
                          6.00644594e-03,
                          1.46498682e-04,
                          1.46498682e-04,
                          7.85379432e-01],
                         [2.09473476e-01, 6.03082090e-02, 4.93924726e-05, 4.93924726e-05,
                          7.95218809e-03]],
                    "30:35": [[0.0008881, 0.0008881, 0.0008881, 0.0008881, 0.0008881],
                              [0.00054171, 0.00054171, 0.01137595, 0.00054171, 0.00054171],
                              [0.00052466, 0.00052466, 0.00052466, 0.00052466, 0.00052466],
                              [0.00039904, 0.00039904, 0.00837989, 0.00039904, 0.00039904],
                              [0.00055991, 0.00055991, 0.01175812, 0.00055991, 0.00055991]],
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
        with open("../../data/WSJ_24.pos", 'r') as f:
            testing_corpus = f.readlines()
        with open("../../data/hmm_vocab.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, prep = preprocess(vocab, "../../data/test.words")
        emission_counts, transition_counts, tag_counts = tagging.create_dictionaries(testing_corpus, vocab)
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
        with open("../../data/WSJ_24.pos", 'r') as f:
            testing_corpus = f.readlines()
        with open("../../data/hmm_vocab.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, corpus = preprocess(vocab, "../../data/test.words")
        emission_counts, transition_counts, tag_counts = tagging.create_dictionaries(testing_corpus, vocab)
        A = tagging.create_transition_matrix(0.001, tag_counts, transition_counts)
        B = tagging.create_emission_matrix(0.001, tag_counts, emission_counts, vocab)
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

    def test_viterbi_forward(self):
        # todo: update tests after validation
        tagging = Tagging()
        target = tagging.viterbi_forward
        with open("../../data/WSJ_02-21.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/hmm_vocab.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, test_corpus = preprocess(vocab, "../../data/test.words")
        emission_counts, transition_counts, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
        A = tagging.create_transition_matrix(0.001, tag_counts, transition_counts)
        B = tagging.create_emission_matrix(0.001, tag_counts, emission_counts, vocab)
        test_cases = [
            {
                "name": "default_check",
                "input": {
                    "A": A,
                    "B": B,
                    "test_corpus": test_corpus,
                    "best_probs": pickle.load(
                        open("../../support_files/best_probs_initilized.pkl", "rb")
                    ),
                    "best_paths": pickle.load(
                        open("../../support_files/best_paths_initilized.pkl", "rb")
                    ),
                    "vocab": vocab,
                    "verbose": False,
                },
                "expected": {
                    "best_probs0:5": np.array(
                        [
                            [
                                -22.60982633,
                                -24.78215633,
                                -34.08246498,
                                -34.34107105,
                                -49.56012613,
                            ],
                            [
                                -23.07660654,
                                -24.51583896,
                                -35.04774303,
                                -35.28281026,
                                -50.52540418,
                            ],
                            [
                                -23.57298822,
                                -29.98305064,
                                -31.98004656,
                                -38.99187549,
                                -47.45770771,
                            ],
                            [
                                -19.76726066,
                                -25.7122143,
                                -31.54577612,
                                -37.38331695,
                                -47.02343727,
                            ],
                            [
                                -24.74325104,
                                -28.78696025,
                                -31.458494,
                                -36.00456711,
                                -46.93615515,
                            ],
                        ]
                    ),
                    "best_probs30:35": np.array(
                        [
                            [
                                -202.75618827,
                                -208.38838519,
                                -210.46938402,
                                -210.15943098,
                                -223.79223672,
                            ],
                            [
                                -202.58297597,
                                -217.72266765,
                                -207.23725672,
                                -215.529735,
                                -224.13957203,
                            ],
                            [
                                -202.00878092,
                                -214.23093833,
                                -217.41021623,
                                -220.73768708,
                                -222.03338753,
                            ],
                            [
                                -200.44016117,
                                -209.46937757,
                                -209.06951664,
                                -216.22297765,
                                -221.09669653,
                            ],
                            [
                                -208.74189499,
                                -214.62088817,
                                -209.79346523,
                                -213.52623459,
                                -228.70417526,
                            ],
                        ]
                    ),
                    "best_paths0:5": np.array(
                        [
                            [0, 11, 20, 25, 20],
                            [0, 11, 20, 25, 20],
                            [0, 11, 20, 25, 20],
                            [0, 11, 20, 25, 20],
                            [0, 11, 20, 25, 20],
                        ]
                    ),
                    "best_paths30:35": np.array(
                        [
                            [20, 19, 35, 11, 21],
                            [20, 19, 35, 11, 21],
                            [20, 19, 35, 11, 21],
                            [20, 19, 35, 11, 21],
                            [35, 19, 35, 11, 34],
                        ]
                    ),
                },
            }
        ]
        for test_case in test_cases:
            result_best_probs, result_best_paths = target(**test_case["input"])
            self.assertEqual(True, isinstance(result_best_probs, np.ndarray))
            self.assertEqual(True, isinstance(result_best_paths, np.ndarray))
            self.assertEqual(True, np.allclose(result_best_probs[0:5, 0:5], test_case["expected"]["best_probs0:5"]))
            self.assertEqual(True,
                             np.allclose(result_best_probs[30:35, 30:35], test_case["expected"]["best_probs30:35"], ))
            self.assertEqual(True, np.allclose(result_best_paths[0:5, 0:5], test_case["expected"]["best_paths0:5"], ))
            self.assertEqual(True,
                             np.allclose(result_best_paths[30:35, 30:35], test_case["expected"]["best_paths30:35"]))


if __name__ == '__main__':
    unittest.main()
