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

    def test_initialize2(self):
        tagging = Tagging()
        with open("../../data/small/tag_small.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/small/vocab_small.txt", 'r') as f:
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
        with open("../../data/small/tag_small.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/small/vocab_small.txt", 'r') as f:
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

    def test_viterbi_backward1(self):
        tagging = Tagging()
        target = tagging.viterbi_backward
        with open("../../data/large/WSJ_02-21.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/large/hmm_vocab.txt", 'r') as f:
            voc_l = f.read().split('\n')
        vocab = {}
        for i, word in enumerate(sorted(voc_l)):
            vocab[word] = i
        _, corpus = preprocess(vocab, "../../data/large/test.words")
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

    def test_viterbi_backward2(self):
        tagging = Tagging()
        with open("../../data/small/tag_small.pos", 'r') as f:
            training_corpus = f.readlines()
        with open("../../data/small/vocab_small.txt", 'r') as f:
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
        with open("../../data/small/tag_small.pos", 'r') as f:
            training_corpus = true_labels = f.readlines()
        with open("../../data/small/vocab_small.txt", 'r') as f:
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
