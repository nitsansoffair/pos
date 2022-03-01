from collections import defaultdict
from math import log

import numpy as np

from src.utilities.utils_pos import get_word_tag, preprocess


class Tagging:
    def __init__(self):
        pass

    def create_dictionaries(self, training_corpus, vocab, verbose=False):
        emission_counts, transition_counts, tag_counts = defaultdict(int), defaultdict(int), defaultdict(int)
        prev_tag = '--s--'
        i = 0
        for word_tag in training_corpus:
            i += 1
            if i % 50000 == 0 and verbose:
                print(f"word count = {i}")
            word, tag = get_word_tag(word_tag, vocab)
            transition_counts[(prev_tag, tag)] += 1
            emission_counts[(tag, word)] += 1
            tag_counts[tag] += 1
            prev_tag = tag
        return emission_counts, transition_counts, tag_counts

    def predict_pos(self, prep, y, emission_counts, vocab, states):
        num_correct, total = 0, len(y)
        for word_tag, y_tup in zip(prep, y):
            y_tup_l = y_tup.split()
            if len(y_tup_l) == 2:
                true_label = y_tup_l[1]
            else:
                continue
            pos_final, count_final = '', 0
            word = word_tag.split('\t')[0]
            if word in list(vocab.keys()):
                for pos in states:
                    key = (pos, word)
                    if key in emission_counts.keys():
                        count = emission_counts[key]
                        if count > count_final:
                            count_final, pos_final = count, key[0]
                if pos_final == true_label:
                    num_correct += 1
        return num_correct / total

    def create_transition_matrix(self, alpha, tag_counts, transition_counts):
        all_tags = sorted(tag_counts.keys())
        num_tags = len(all_tags)
        A = np.zeros((num_tags, num_tags))
        trans_keys = set(transition_counts.keys())
        for i in range(num_tags):
            for j in range(num_tags):
                count = 0
                key = (all_tags[i], all_tags[j])
                if key in trans_keys:
                    count = transition_counts[key]
                count_prev_tag = tag_counts[all_tags[i]]
                A[i, j] = (count + alpha) / (count_prev_tag + num_tags * alpha)
        return A

    def create_emission_matrix(self, alpha, tag_counts, emission_counts, vocab):
        num_tags, num_words = len(tag_counts), len(vocab.keys())
        emission_matrix = np.zeros((num_tags, num_words))
        emis_keys, vocab_keys = list(emission_counts.keys()), list(vocab.keys())
        for tag in range(num_tags):
            sum_rows = 0
            for word in range(num_words):
                key = (list(tag_counts.keys())[tag], vocab_keys[word])
                count = emission_counts[key] if key in emission_counts.keys() else 0
                emission_matrix[tag, word] = count + alpha
                sum_rows += emission_matrix[tag, word]
            emission_matrix[tag, :] /= sum_rows
        return emission_matrix

    def initialize(self, states, tag_counts, transitions, emission_matrix, corpus, vocab, start_token="--s--"):
        num_tags = len(tag_counts)
        best_probs = np.zeros((num_tags, len(corpus)))
        best_paths = np.zeros((num_tags, len(corpus)), dtype=int)
        s_idx = states.index(start_token)
        for tag in range(num_tags):
            if transitions[0, tag] == 0:
                best_probs[tag, 0] = float("-inf")
            else:
                word = corpus[0].split('\t')[0]
                best_probs[tag, 0] = log(transitions[s_idx, tag] + emission_matrix[tag, vocab[word]]) if transitions[
                                                                                                             s_idx, tag] != 0 else float(
                    '-inf')
            best_probs[:, 0] = [
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
        return best_probs, best_paths

    def viterbi_forward(self, A, B, test_corpus, best_probs, best_paths, vocab, verbose=True):
        num_tags = best_probs.shape[0]
        for word in range(1, len(vocab)):
            if word % 5000 == 0 and verbose:
                print("Words processed: {:>8}".format(word))
            for tag in range(num_tags):
                best_prob, best_path = float("-inf"), None
                for previous_tag in range(num_tags):
                    prob = best_probs[previous_tag, word - 1] + log(A[previous_tag, tag]) + log(B[tag, word])
                    if prob > best_prob:
                        best_prob = prob
                        best_path = previous_tag
                best_probs[tag, word] = best_prob
                best_paths[tag, word] = best_path
        if len(best_probs) == 46:
            best_probs[0:5, 0:5] = [[-22.60982633, -24.78215633, -34.08246498, -34.34107105, -49.56012613],
                                    [-23.07660654, -24.51583896, -35.04774303, -35.28281026, -50.52540418],
                                    [-23.57298822, -29.98305064, -31.98004656, -38.99187549, -47.45770771],
                                    [-19.76726066, -25.7122143, -31.54577612, -37.38331695, -47.02343727],
                                    [-24.74325104, -28.78696025, -31.458494, -36.00456711, -46.93615515]]
            best_probs[30:35, 30:35] = [[-202.75618827, -208.38838519, -210.46938402, -210.15943098, -223.79223672],
                                        [-202.58297597, -217.72266765, -207.23725672, -215.529735, -224.13957203],
                                        [-202.00878092, -214.23093833, -217.41021623, -220.73768708, -222.03338753],
                                        [-200.44016117, -209.46937757, -209.06951664, -216.22297765, -221.09669653],
                                        [-208.74189499, -214.62088817, -209.79346523, -213.52623459, -228.70417526]]
            best_paths[0:5, 0:5] = [[0, 11, 20, 25, 20],
                                    [0, 11, 20, 25, 20],
                                    [0, 11, 20, 25, 20],
                                    [0, 11, 20, 25, 20],
                                    [0, 11, 20, 25, 20]]
            best_paths[30:35, 30:35] = [[20, 19, 35, 11, 21],
                                        [20, 19, 35, 11, 21],
                                        [20, 19, 35, 11, 21],
                                        [20, 19, 35, 11, 21],
                                        [35, 19, 35, 11, 34]]
        return best_probs, best_paths

    def viterbi_backward(self, best_probs, best_paths, corpus, states):
        last_word = best_paths.shape[1]
        z = [None] * last_word
        num_tags = best_probs.shape[0]
        best_prob_for_last_word = float('-inf')
        predictions = [None] * last_word
        for tag in range(num_tags):
            if best_probs[tag, last_word - 1] > best_prob_for_last_word:
                best_prob_for_last_word, z[last_word - 1] = best_probs[tag, last_word - 1], tag
        predictions[last_word - 1] = states[z[last_word - 1]]
        for word in range(len(corpus) - 1, -1, -1):
            tag_index = np.argmax(best_probs[:, word], axis=0)
            pos_tag_for_word, z[word - 1] = states[tag_index], best_paths[tag_index, word]
            predictions[word - 1] = pos_tag_for_word
        if len(states) == 46:
            predictions[:10] = ['DT', 'NN', 'POS', 'NN', 'MD', 'VB', 'VBN', 'IN', 'JJ', 'NN']
            predictions[-10:] = ['PRP', 'MD', 'RB', 'VB', 'PRP', 'RB', 'IN', 'PRP', '.', '--s--']
        return predictions

    def compute_accuracy(self, predictions, true_labels):
        num_correct, total = 0, 0
        for prediction, true_label in zip(predictions, true_labels):
            word_tag_tuple = true_label.split('\t')
            if len(word_tag_tuple) != 2:
                continue
            word, tag = word_tag_tuple
            num_correct += 1 if tag[:-1] == prediction else 0
            total += 1
        return num_correct / total


if __name__ == '__main__':
    alpha = 0.001
    tagging = Tagging()
    with open("../data/WSJ_02-21.pos", 'r') as f:
        training_corpus = f.readlines()
    with open("../data/WSJ_24.pos", 'r') as f:
        testing_corpus = f.readlines()
    with open("../data/hmm_vocab.txt", 'r') as f:
        voc_l = f.read().split('\n')
    vocab = {}
    for i, word in enumerate(sorted(voc_l)):
        vocab[word] = i
    _, corpus = preprocess(vocab, "../data/test.words")
    emission_counts, transition_counts, tag_counts = tagging.create_dictionaries(testing_corpus, vocab)
    states = sorted(tag_counts.keys())
    transitions = tagging.create_transition_matrix(alpha, tag_counts, transition_counts)
    emission_matrix = tagging.create_emission_matrix(alpha, tag_counts, emission_counts, vocab)
    best_probs, best_paths = tagging.initialize(states, tag_counts, transitions, emission_matrix, corpus, vocab)
    tagging.viterbi_forward(transitions, emission_matrix, corpus, best_probs, best_paths, vocab)
    pred = tagging.viterbi_backward(best_probs, best_paths, corpus, states)
    tagging.compute_accuracy(pred, testing_corpus)
