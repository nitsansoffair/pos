from collections import defaultdict

import numpy as np
import pandas as pd

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
        for word, y_tup in zip(prep, y):
            y_tup_l = y_tup.split()
            if len(y_tup_l) == 2:
                true_label = y_tup_l[1]
            else:
                continue
            pos_final, count_final = '', 0
            if word in vocab:
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
        # todo: validate
        tags_keys, total_tags = list(tag_counts.keys()), len(tag_counts.keys())
        transitions = np.zeros((total_tags, total_tags))
        for source in range(total_tags):
            for target in range(total_tags):
                key = (tags_keys[source], tags_keys[target])
                count = transition_counts[key] if key in transition_counts.keys() else 0
                transitions[source, target] = (count + alpha) / (tag_counts[key[0]] + alpha * total_tags)
        return transitions

if __name__ == '__main__':
    alpha = 0.001
    tagging = Tagging()
    with open("../data/WSJ_02-21.pos", 'r') as f:
        training_corpus = f.readlines()
    with open("../data/hmm_vocab.txt", 'r') as f:
        voc_l = f.read().split('\n')
    vocab = {}
    for i, word in enumerate(sorted(voc_l)):
        vocab[word] = i
    emission_counts, transition_counts, tag_counts = tagging.create_dictionaries(training_corpus, vocab)
    states = sorted(tag_counts.keys())
    transitions = tagging.create_transition_matrix(alpha, tag_counts, transition_counts)
    transitions_sub = pd.DataFrame(transitions[30:35, 30:35], index=states[30:35], columns=states[30:35])