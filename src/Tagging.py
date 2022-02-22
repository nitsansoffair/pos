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
        tags_keys, total_tags = list(tag_counts.keys()), len(tag_counts.keys())
        transitions = np.zeros((total_tags, total_tags))
        for source in range(total_tags):
            row_sum = 0
            for target in range(total_tags):
                key = (tags_keys[source], tags_keys[target])
                count = transition_counts[key] if key in transition_counts.keys() else 0
                transitions[source, target] = count + alpha
                row_sum += transitions[source, target]
            transitions[source, :] /= row_sum
        return transitions

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

    def initialize(self, states, tag_counts, A, B, corpus, vocab):
        num_tags = len(tag_counts)
        best_probs = np.zeros((num_tags, len(corpus)))
        best_paths = np.zeros((num_tags, len(corpus)), dtype=int)
        s_idx = states.index("--s--")
        for i in range(num_tags):
            if A[0, i] == 0:
                best_probs[i, 0] = float("-inf")
            else:
                best_probs[i, 0] = log(A[s_idx, i] + B[i, vocab[corpus[0]]]) if A[s_idx, i] != 0 else float('-inf')
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
        return best_probs, best_paths

    def viterbi_backward(self, best_probs, best_paths, corpus, states):
        last = best_paths.shape[1]
        z = [None] * last
        num_tags = best_probs.shape[0]
        best_prob_for_last_word = float('-inf')
        pred = [None] * last
        for tag in range(num_tags):
            if best_probs[tag, last - 1] > best_prob_for_last_word:
                best_prob_for_last_word = best_probs[tag, last - 1]
                z[last - 1] = tag
        pred[last - 1] = states[z[last - 1]]
        for word in range(len(corpus) - 1, 0, -1):
            pos_tag_for_word = np.argmax(best_probs[:, word], axis=0)
            z[word - 1] = best_paths[pos_tag_for_word, word]
            pred[word - 1] = states[pos_tag_for_word]
        return pred

    def compute_accuracy(self, pred, y):
        num_correct = 0
        total = 0
        for prediction, y in zip(pred, y):
            word_tag_tuple = y.split('\t')
            if len(word_tag_tuple) != 2:
                continue
            word, tag = word_tag_tuple
            if tag[:-1] == prediction:
                num_correct += 1
            total += 1
        return num_correct / total

if __name__ == '__main__':
    alpha = 0.001
    tagging = Tagging()
    with open("../data/large/WSJ_02-21.pos", 'r') as f:
        training_corpus = f.readlines()
    with open("../data/large/WSJ_24.pos", 'r') as f:
        testing_corpus = f.readlines()
    with open("../data/large/hmm_vocab.txt", 'r') as f:
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