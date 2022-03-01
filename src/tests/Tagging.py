import pickle
import unittest
from collections import defaultdict

import numpy as np
import pandas as pd

from src.utilities.utils_pos import preprocess

from src.Tagging import Tagging


class TaggingTest(unittest.TestCase):
    def test_create_dictionaries1(self):
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

    def test_create_dictionaries2(self):
        tagging = Tagging()
        with open("../../data/others/small/tag_small.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/others/small/vocab_small.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        result_emission, result_transition, result_tag = tagging.create_dictionaries(training_corpus, vocab)
        self.assertEqual(list(result_tag.keys()), ['NN', 'IN', 'DT', '.', 'CC', 'JJ', 'VBZ', 'VBN', 'TO', 'NNS', 'VB'])
        self.assertEqual(list(result_tag.values()), [4, 3, 3, 1, 1, 3, 1, 1, 2, 1, 1])
        self.assertEqual(list(result_transition.keys()), [('--s--', 'NN'), ('NN', 'IN'), ('IN', 'DT'), ('DT', 'NN'),
                                                          ('NN', '.'), ('.', 'CC'), ('CC', 'DT'), ('DT', 'JJ'),
                                                          ('JJ', 'NN'),
                                                          ('NN', 'VBZ'), ('VBZ', 'VBN'), ('VBN', 'TO'), ('TO', 'NNS'),
                                                          ('NNS', 'TO'), ('TO', 'VB'), ('VB', 'JJ'), ('JJ', 'IN')])
        self.assertEqual(list(result_transition.values()), [1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_predict_pos1(self):
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
        _, prep = preprocess(vocab, "../../data/large/test.words")
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

    def test_predict_pos2(self):
        tagging = Tagging()
        with open("../../data/others/small/tag_small.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/others/small/vocab_small.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        emission_counts, _, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
        prediction = tagging.predict_pos(training_corpus, training_corpus, emission_counts, vocab,
                                         sorted(tag_counts.keys()))
        self.assertEqual(prediction, 1)

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
            for row in range(5):
                for column in range(5):
                    self.assertEqual(True, abs(test_case["expected"]["0:5"][row, column] - result[0:5, 0:5][row, column]) < .1)
            self.assertEqual(True, np.allclose(result[30:35, 30:35], test_case["expected"]["30:35"]))

    def test_create_emission_matrix(self):
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
            for row in range(5):
                for column in range(5):
                    self.assertEqual(True, abs(test_case["expected"]["0:5"][row, column] - result[0:5, 0:5][row, column]) < .1)
                    self.assertEqual(True, abs(result[30:35, 30:35][row, column] - test_case["expected"]["30:35"][row, column]) < .1)

    def test_initialize(self):
        tagging = Tagging()
        target = tagging.initialize
        with open("../../data/WSJ_02-21.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/hmm_vocab.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        emission_counts, transition_counts, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
        states = sorted(tag_counts.keys())
        _, corpus = preprocess(vocab, "../../data/test.words")
        alpha = 0.001
        A = tagging.create_transition_matrix(alpha, tag_counts, transition_counts)
        B = tagging.create_emission_matrix(alpha, tag_counts, emission_counts, vocab)
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

    def test_initialize2(self):
        tagging = Tagging()
        with open("../../data/others/small/tag_small.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/others/small/vocab_small.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, corpus = preprocess(vocab, "../../data/small/train.words")
        emission_counts, transition_counts, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
        transitions = tagging.create_transition_matrix(0.001, tag_counts, transition_counts)
        emission_matrix = tagging.create_emission_matrix(0.001, tag_counts, emission_counts, vocab)
        states = tags = list(tag_counts.keys())
        best_probs, best_paths = tagging.initialize(states, tag_counts, transitions, emission_matrix, training_corpus,
                                                    vocab,
                                                    start_token="NN")
        words = []
        for tuple in training_corpus:
            words.append(tuple.split('\t')[0])
        df_probs = pd.DataFrame(best_probs[:, :1], index=tags[:], columns=words[:1])
        true_probs = [[-7.60452014],
                      [-0.69472957],
                      [-7.45134578],
                      [-1.38411267],
                      [-6.69941463],
                      [-7.45134578],
                      [-1.38411267],
                      [-6.69941463],
                      [-7.20232163],
                      [-6.69941463],
                      [-0.0165869]]
        for row in range(df_probs.shape[1]):
            self.assertEqual(True, abs(true_probs[row][0] - best_probs[row, 0]) < .01)
        self.assertEqual(0, np.sum(best_paths[:, :1], axis=0))

    def test_viterbi_forward(self):
        tagging = Tagging()
        target = tagging.viterbi_forward
        alpha = 0.001
        with open("../../data/WSJ_02-21.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/hmm_vocab.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, test_corpus = preprocess(vocab, "../../data/test.words")
        emission_counts, transition_counts, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
        A = tagging.create_transition_matrix(alpha, tag_counts, transition_counts)
        B = tagging.create_emission_matrix(alpha, tag_counts, emission_counts, vocab)
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
            self.assertEqual(True, np.allclose(
                result_best_probs[0:5, 0:5], test_case["expected"]["best_probs0:5"]
            ))
            self.assertEqual(True, np.allclose(
                result_best_probs[30:35, 30:35],
                test_case["expected"]["best_probs30:35"],
            ))
            self.assertEqual(True, np.allclose(
                result_best_paths[0:5, 0:5], test_case["expected"]["best_paths0:5"],
            ))
            self.assertEqual(True, np.allclose(
                result_best_paths[30:35, 30:35],
                test_case["expected"]["best_paths30:35"],
            ))

    def test_viterbi_forward2(self):
        tagging = Tagging()
        with open("../../data/others/small/tag_small.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/others/small/vocab_small.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, train_corpus = preprocess(vocab, "../../data/small/train.words")
        emission_counts, transition_counts, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
        transitions = tagging.create_transition_matrix(0.001, tag_counts, transition_counts)
        emission_matrix = tagging.create_emission_matrix(0.001, tag_counts, emission_counts, vocab)
        states = tags = list(tag_counts.keys())
        words = []
        for tuple in training_corpus:
            words.append(tuple.split('\t')[0])
        best_probs, best_paths = tagging.initialize(states, tag_counts, transitions, emission_matrix, training_corpus,
                                                    vocab,
                                                    start_token="NN")
        best_probs, best_paths, tagging.viterbi_forward(transitions, emission_matrix, train_corpus,
                                                        best_probs, best_paths, vocab)
        df_probs = pd.DataFrame(best_probs[:, :3], index=tags[:], columns=words[:3])
        df_paths = pd.DataFrame(best_paths[:, :3], index=tags[:], columns=words[:3])
        true_probs = [[-7.60452014e+00, -1.52338217e+01, -1.67460417e+01],
                      [-6.94729571e-01, -1.49476318e+01, -1.71524994e+01],
                      [-7.45134578e+00, -8.71206427e+00, -1.63419384e+01],
                      [-1.38411267e+00, -6.95212254e+00, -2.07964130e+01],
                      [-6.69941463e+00, -8.31964831e+00, -6.97890340e+00],
                      [-7.45134578e+00, -8.03887698e+00, -1.71330391e+01],
                      [-1.38411267e+00, -1.38608773e+01, -2.07964130e+01],
                      [-6.69941463e+00, -8.31964831e+00, -2.07964130e+01],
                      [-7.20232163e+00, -1.43192173e+01, -1.59394509e+01],
                      [-6.69941463e+00, -1.38608773e+01, -2.07964130e+01],
                      [-1.65869014e-02, -1.38608773e+01, -2.07964130e+01]]
        true_paths = [[0, 10, 5],
                      [0, 10, 5],
                      [0, 1, 4],
                      [0, 10, 3],
                      [0, 3, 3],
                      [0, 10, 2],
                      [0, 10, 3],
                      [0, 6, 3],
                      [0, 7, 7],
                      [0, 10, 3],
                      [0, 10, 3]]
        for row in range(best_probs.shape[0]):
            for column in range(3):
                self.assertEqual(True, abs(best_probs[row, column] - true_probs[row][column]) < .01)
                self.assertEqual(True, abs(best_paths[row, column] - true_paths[row][column]) < .01)

    def test_viterbi_backward(self):
        tagging = Tagging()
        target = tagging.viterbi_backward
        with open("../../data/WSJ_02-21.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/hmm_vocab.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        emission_counts, transition_counts, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
        states = sorted(tag_counts.keys())
        _, corpus = preprocess(vocab, "../../data/test.words")
        test_cases = [
            {
                "name": "default_check",
                "input": {
                    "corpus": corpus,
                    "best_probs": pickle.load(
                        open("../../support_files/best_probs_trained.pkl", "rb")
                    ),
                    "best_paths": pickle.load(
                        open("../../support_files/best_paths_trained.pkl", "rb")
                    ),
                    "states": states,
                },
                "expected": {
                    "pred_len": 34199,
                    "pred_head": [
                        "DT",
                        "NN",
                        "POS",
                        "NN",
                        "MD",
                        "VB",
                        "VBN",
                        "IN",
                        "JJ",
                        "NN",
                    ],
                    "pred_tail": [
                        "PRP",
                        "MD",
                        "RB",
                        "VB",
                        "PRP",
                        "RB",
                        "IN",
                        "PRP",
                        ".",
                        "--s--",
                    ],
                },
            }
        ]
        for test_case in test_cases:
            result = target(**test_case["input"])
            self.assertEqual(True, isinstance(result, list))
            self.assertEqual(True, len(result) == test_case["expected"]["pred_len"])
            self.assertEqual(test_case["expected"]["pred_head"], result[:10])
            self.assertEqual(test_case["expected"]["pred_tail"], result[-10:])

    def test_viterbi_backward2(self):
        tagging = Tagging()
        with open("../../data/others/small/tag_small.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/others/small/vocab_small.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, train_corpus = preprocess(vocab, "../../data/small/train.words")
        emission_counts, transition_counts, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
        transitions = tagging.create_transition_matrix(0.001, tag_counts, transition_counts)
        emission_matrix = tagging.create_emission_matrix(0.001, tag_counts, emission_counts, vocab)
        states = tags = list(tag_counts.keys())
        words = []
        for tuple in training_corpus:
            words.append(tuple.split('\t')[0])
        best_probs, best_paths = tagging.initialize(states, tag_counts, transitions, emission_matrix, training_corpus,
                                                    vocab,
                                                    start_token="NN")
        best_probs, best_paths, tagging.viterbi_forward(transitions, emission_matrix, train_corpus,
                                                        best_probs, best_paths, vocab)
        predictions = tagging.viterbi_backward(best_probs, best_paths, train_corpus, states)
        true_predictions = ['NN',
                            'IN',
                            'DT',
                            'NN',
                            '.',
                            'CC',
                            'DT',
                            'JJ',
                            'NN',
                            'IN',
                            'DT',
                            'JJ',
                            'NN',
                            'VBZ',
                            'VBN',
                            'TO',
                            'NNS',
                            'TO',
                            'VB',
                            'JJ',
                            'IN']
        total_trues = 0
        for index in range(len(predictions)):
            if predictions[index] == true_predictions[index]:
                total_trues += 1
        self.assertEqual(total_trues, len(predictions))

    def test_compute_accuracy(self):
        tagging = Tagging()
        with open("../../data/others/small/tag_small.pos", 'r') as f:
            training_corpus = true_labels = f.readlines()
        with open("../../data/others/small/vocab_small.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, train_corpus = preprocess(vocab, "../../data/small/train.words")
        emission_counts, transition_counts, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
        transitions = tagging.create_transition_matrix(0.001, tag_counts, transition_counts)
        emission_matrix = tagging.create_emission_matrix(0.001, tag_counts, emission_counts, vocab)
        states = tags = list(tag_counts.keys())
        words = []
        for tuple in training_corpus:
            words.append(tuple.split('\t')[0])
        best_probs, best_paths = tagging.initialize(states, tag_counts, transitions, emission_matrix, training_corpus,
                                                    vocab,
                                                    start_token="NN")
        best_probs, best_paths, tagging.viterbi_forward(transitions, emission_matrix, train_corpus,
                                                        best_probs, best_paths, vocab)
        predictions = tagging.viterbi_backward(best_probs, best_paths, train_corpus, states)
        accuracy = tagging.compute_accuracy(predictions, true_labels)
        self.assertEqual(True, 0 <= accuracy <= 1)
        self.assertEqual(True, accuracy > .95)


if __name__ == '__main__':
    unittest.main()
