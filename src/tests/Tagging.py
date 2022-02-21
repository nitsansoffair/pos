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
        with open("../../data/large/WSJ_02-21.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/large/hmm_vocab.txt", 'r') as f:
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

    def test_create_dictionaries_small(self):
        tagging = Tagging()
        with open("../../data/small/tag_small.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/small/vocab_small.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        result_emission, result_transition, result_tag = tagging.create_dictionaries(training_corpus, vocab)
        self.assertEqual(list(result_tag.keys()), ['NN', 'IN', 'DT', '.', 'CC', 'JJ', 'VBZ', 'VBN', 'TO', 'NNS', 'VB'])
        self.assertEqual(list(result_tag.values()), [4, 3, 3, 1, 1, 3, 1, 1, 2, 1, 1])
        self.assertEqual(list(result_transition.keys()), [('--s--', 'NN'), ('NN', 'IN'), ('IN', 'DT'), ('DT', 'NN'),
                                                          ('NN', '.'), ('.', 'CC'), ('CC', 'DT'), ('DT', 'JJ'), ('JJ', 'NN'),
                                                          ('NN', 'VBZ'), ('VBZ', 'VBN'), ('VBN', 'TO'), ('TO', 'NNS'),
                                                          ('NNS', 'TO'), ('TO', 'VB'), ('VB', 'JJ'), ('JJ', 'IN')])
        self.assertEqual(list(result_transition.values()), [1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_predict_pos(self):
        tagging = Tagging()
        target = tagging.predict_pos
        with open("../../data/large/WSJ_02-21.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/large/hmm_vocab.txt", 'r') as f:
            voc_l = f.read().split('\n')
        with open("../../data/large/WSJ_24.pos", 'r') as f:
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

    def test_predict_pos_small(self):
        tagging = Tagging()
        with open("../../data/small/tag_small.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/small/vocab_small.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        emission_counts, _, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
        prediction = tagging.predict_pos(training_corpus, training_corpus, emission_counts, vocab, sorted(tag_counts.keys()))
        self.assertEqual(prediction, 1)

    def test_create_transition_matrix(self):
        tagging = Tagging()
        target = tagging.create_transition_matrix
        with open("../../data/large/WSJ_02-21.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/large/hmm_vocab.txt", 'r') as f:
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
            # todo: fix those tests
            # self.assertEqual(True, np.allclose(result[0:5, 0:5], test_case["expected"]["0:5"]))
            # self.assertEqual(True, np.allclose(result[30:35, 30:35], test_case["expected"]["30:35"]))

    def test_create_emission_matrix(self):
        tagging = Tagging()
        target = tagging.create_emission_matrix
        with open("../../data/large/WSJ_02-21.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/large/hmm_vocab.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, prep = preprocess(vocab, "../../data/test.words")
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
            # todo: fix those tests
            # self.assertEqual(True, np.allclose(result[0:5, 0:5], test_case["expected"]["0:5"]))
            # self.assertEqual(True, np.allclose(result[30:35, 30:35], test_case["expected"]["30:35"]))

    def test_initialize(self):
        tagging = Tagging()
        target = tagging.initialize
        with open("../../data/large/WSJ_02-21.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/large/hmm_vocab.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, corpus = preprocess(vocab, "../../data/test.words")
        emission_counts, transition_counts, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
        A = tagging.create_transition_matrix(0.001, tag_counts, transition_counts)
        B = tagging.create_emission_matrix(0.001, tag_counts, emission_counts, vocab)
        states = list(tag_counts.keys())
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
                    "best_probs_col0": [-1.31597142, -3.37625714, -12.46976978, -5.59522983, -5.12489652,
                                        -5.25868772, -1.96847024, -3.31307842, -3.21591202, -13.85426655,
                                        -2.69404906, -2.59947745, -13.85426655, -7.20313406, -1.77997063,
                                        -6.10576485, -5.41293887, -13.85426655, -4.90221211, -4.43225704,
                                        -2.65102354, -7.20363265, -2.76225203, -4.90214334, -6.49603028,
                                        -4.56575126, -2.66161813, -4.71962758, -2.58952989, -4.12153317,
                                        -13.85426655, -5.59522983, -6.10576338, -6.10585776, -13.85426655,
                                        -13.85426655, -7.20207357, -6.10540755, -5.25864173, -13.27887493,
                                        -6.10585776, -12.22353249, -4.90211673, -13.3022365, -7.38841061,
                                        -5.81807025],
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
        tagging = Tagging()
        target = tagging.viterbi_forward
        with open("../../data/large/WSJ_02-21.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/large/hmm_vocab.txt", 'r') as f:
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
                    "best_probs0:5": [[-22.60982633, -24.81569994, -40.98975291, -53.25491965, -70.17179432],
                                      [-23.07660654, -24.37453926, -41.01527222, -52.66763772, -71.66260044],
                                      [-23.57298822, -26.38711047, -43.54969854, -53.90344415, -72.73173994],
                                      [-19.76726066, -26.13694219, -43.28506262, -52.8003639, -71.1596961],
                                      [-24.74325104, -26.22230661, -43.01562103, -54.81607812, -56.70642324]],
                    "best_probs30:35": [[-372.8198332, -379.28984239, -386.36087011, -398.20780805, -400.67550792],
                                        [-370.24027947, -372.32981343, -383.96731574, -395.54277982, -408.25047519],
                                        [-372.66124123, -379.364461, -387.01527579, -397.4392292, -410.4625349],
                                        [-374.89370194, -379.28984239, -389.57878644, -397.07701182, -402.16668669],
                                        [-374.30182204, -373.14523287, -387.07530008, -398.8498702, -413.00607316]],
                    "best_paths0:5": [[0, 11, 9, 15, 9],
                                      [0, 11, 40, 15, 9],
                                      [0, 11, 9, 15, 9],
                                      [0, 11, 16, 15, 37],
                                      [0, 11, 9, 15, 1]],
                    "best_paths30:35": [[12, 41, 29, 4, 4],
                                        [12, 41, 29, 4, 4],
                                        [15, 41, 29, 35, 27],
                                        [15, 41, 29, 35, 27],
                                        [15, 41, 29, 20, 20]],
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

    def test_viterbi_backward(self):
        tagging = Tagging()
        target = tagging.viterbi_backward
        with open("../../data/large/WSJ_02-21.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/large/hmm_vocab.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, corpus = preprocess(vocab, "../../data/test.words")
        _, _, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
        states = list(tag_counts.keys())
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
                    "pred_head": ['VBZ', 'JJ', 'VBZ', '--s--', 'WRB', 'RP', ')', 'VBD', 'PRP$', 'PRP$'],
                    "pred_tail": ['--s--', 'RBR', 'WRB', 'MD', 'RBR', ')', 'MD', 'POS', "''", "''"],
                },
            }
        ]
        for test_case in test_cases:
            result = target(**test_case["input"])
            self.assertEqual(True, isinstance(result, list))
            self.assertEqual(True, len(result) == test_case["expected"]["pred_len"])
            self.assertEqual(True, result[:10] == test_case["expected"]["pred_head"])
            self.assertEqual(True, result[-10:] == test_case["expected"]["pred_tail"])

    def test_compute_accuracy(self):
        tagging = Tagging()
        target = tagging.compute_accuracy
        alpha = 0.001
        with open("../../data/large/WSJ_24.pos", 'r') as f:
            test_data = f.readlines()
        with open("../../data/large/WSJ_02-21.pos", 'r') as f:
            train_data = f.readlines()
        with open("../../data/large/hmm_vocab.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, corpus = preprocess(vocab, "../../data/test.words")
        emission_counts, transition_counts, tag_counts = tagging.create_dictionaries(train_data, vocab)
        states = sorted(tag_counts.keys())
        transitions = tagging.create_transition_matrix(alpha, tag_counts, transition_counts)
        emission_matrix = tagging.create_emission_matrix(alpha, tag_counts, emission_counts, vocab)
        best_probs, best_paths = tagging.initialize(states, tag_counts, transitions, emission_matrix, corpus, vocab)
        tagging.viterbi_forward(transitions, emission_matrix, corpus, best_probs, best_paths, vocab)
        pred = tagging.viterbi_backward(best_probs, best_paths, corpus, states)
        test_cases = [
            {
                "name": "default_check",
                "input": {"pred": pred, "y": test_data},
                "expected": 0.953063647155511,
            },
            {
                "name": "small_check",
                "input": {"pred": pred[:100], "y": test_data[:100]},
                "expected": 0.979381443298969,
            },
        ]
        for test_case in test_cases:
            result = target(**test_case["input"])
            self.assertEqual(True, isinstance(result, float))
            self.assertEqual(True, np.isclose(result, test_case["expected"]))

if __name__ == '__main__':
    unittest.main()
