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
                    "0:5": [[3.47249110e-07, 3.47249110e-07, 3.47249110e-07, 3.47249110e-07,
                             3.47249110e-07],
                            [2.19068752e-07, 2.19068752e-07, 2.19068752e-07, 2.19068752e-07,
                             3.22033256e-02],
                            [3.09811418e-06, 3.09811418e-06, 3.09811418e-06, 3.09811418e-06,
                             3.09811418e-06],
                            [2.19068752e-07, 2.19068752e-07, 2.19068752e-07, 2.19068752e-07,
                             3.22033256e-02],
                            [2.75651433e-06, 2.75651433e-06, 2.75651433e-06, 2.75651433e-06,
                             2.75651433e-06]],
                    "30:35": [[2.19068752e-07, 1.68685130e-02, 2.19068752e-07, 2.19068752e-07,
                               1.31463158e-03],
                              [2.19068752e-07, 1.68685130e-02, 2.19068752e-07, 2.19068752e-07,
                               1.31463158e-03],
                              [4.85489449e-07, 1.89345740e-02, 1.26232112e-02, 4.85489449e-07,
                               1.01957639e-02],
                              [2.75044372e-07, 2.75044372e-07, 2.75044372e-07, 2.75044372e-07,
                               2.75044372e-07],
                              [2.19068752e-07, 1.68685130e-02, 2.19068752e-07, 2.19068752e-07,
                               1.31463158e-03]],
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
                    "0:5": [[1.23613978e-05, 1.23613978e-05, 1.23613978e-05, 1.23613978e-05,
                             1.23613978e-05],
                            [8.72623193e-06, 8.72623193e-06, 8.72623193e-06, 8.72623193e-06,
                             2.56638481e-02],
                            [3.36055382e-05, 3.36055382e-05, 3.36055382e-05, 3.36055382e-05,
                             3.36055382e-05],
                            [8.72623193e-06, 8.72623193e-06, 8.72623193e-06, 8.72623193e-06,
                             2.56638481e-02],
                            [3.27257257e-05, 3.27257257e-05, 3.27257257e-05, 3.27257257e-05,
                             3.27257257e-05]],
                    "30:35": [[8.72623193e-06, 1.34471234e-02, 8.72623193e-06, 8.72623193e-06,
                               1.05587406e-03],
                              [8.72623193e-06, 1.34471234e-02, 8.72623193e-06, 8.72623193e-06,
                               1.05587406e-03],
                              [1.55045971e-05, 1.21090903e-02, 8.07789510e-03, 1.55045971e-05,
                               6.52743538e-03],
                              [1.04148224e-05, 1.04148224e-05, 1.04148224e-05, 1.04148224e-05,
                               1.04148224e-05],
                              [8.72623193e-06, 1.34471234e-02, 8.72623193e-06, 8.72623193e-06,
                               1.05587406e-03]],
                },
            },
        ]
        for test_case in test_cases:
            result = target(**test_case["input"])
            self.assertEqual(True, isinstance(result, np.ndarray))
            self.assertEqual(True, np.allclose(result[0:5, 0:5], test_case["expected"]["0:5"]))
            self.assertEqual(True, np.allclose(result[30:35, 30:35], test_case["expected"]["30:35"]))

    def test_initialize(self):
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


if __name__ == '__main__':
    unittest.main()
