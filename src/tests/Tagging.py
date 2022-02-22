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

    def test_create_dictionaries2(self):
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
                                                          ('NN', '.'), ('.', 'CC'), ('CC', 'DT'), ('DT', 'JJ'),
                                                          ('JJ', 'NN'),
                                                          ('NN', 'VBZ'), ('VBZ', 'VBN'), ('VBN', 'TO'), ('TO', 'NNS'),
                                                          ('NNS', 'TO'), ('TO', 'VB'), ('VB', 'JJ'), ('JJ', 'IN')])
        self.assertEqual(list(result_transition.values()), [1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_predict_pos1(self):
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

    def test_predict_pos2(self):
        tagging = Tagging()
        with open("../../data/small/tag_small.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/small/vocab_small.txt", 'r') as f:
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
        with open("../../data/small/tag_small.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/small/vocab_small.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, transition_counts, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
        transition_matrix = tagging.create_transition_matrix(0.001, tag_counts, transition_counts)
        rows, columns = np.shape(transition_matrix)
        for row in range(rows):
            self.assertEqual(True, abs(1 - np.sum(transition_matrix[row, :], axis=0)) < .01)
        true_matrix = [[2.49314385e-04, 4.98878085e-01, 2.49314385e-04],
                       [4.97265042e-04, 4.97265042e-04, 9.95027350e-01],
                       [3.32447692e-01, 3.32115576e-04, 3.32115576e-04]]
        for row in range(3):
            for column in range(3):
                self.assertEqual(True, abs(transition_matrix[row, column] - true_matrix[row][column]) < .01)

    def test_create_emission_matrix(self):
        tagging = Tagging()
        with open("../../data/small/tag_small.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/small/vocab_small.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, prep = preprocess(vocab, "../../data/small/train.words")
        emission_counts, _, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
        emission_matrix = tagging.create_emission_matrix(.001, tag_counts, emission_counts, vocab)
        self.assertEqual(emission_matrix.shape, (11, 18))
        emission_matrix_cut = emission_matrix[-3:, -3:]
        true_matrix = [[4.95540139e-04, 4.95540139e-04, 9.91575818e-01],
                       [9.82318271e-04, 9.82318271e-04, 9.82318271e-04],
                       [9.82318271e-04, 9.82318271e-04, 9.82318271e-04]]
        for row in range(3):
            for column in range(3):
                self.assertEqual(True, abs(emission_matrix_cut[row, column] - true_matrix[row][column]) < .01)

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
        _, corpus = preprocess(vocab, "../../data/large/test.words")
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
            self.assertEqual(True, result_best_paths.shape == test_case["expected"]["best_paths_shape"])
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
