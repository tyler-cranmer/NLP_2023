import nltk
from nltk.corpus import indian
import numpy as np
import random
from typing import List

nltk.download('indian')

UNK_TOKEN = '<unk>'


def get_observation_ids(observations, ob_dict):
    return [[ob_dict[e] if e in ob_dict else ob_dict[UNK_TOKEN] for e in es] for es in observations]


def get_state_ids(tags, state_dict):
    return [[state_dict[t] for t in ts] for ts in tags]


def generate_hmm_data(tagged_sentences):
    """
    create the state, observation dict from nltk tagged sentences
    each tag sentence is of the form list[(word, pos_tag)]
    """
    words = list(set([word for tagged_sent in tagged_sentences for word, tag in tagged_sent]))
    words.sort()
    tags = list(set([tag for tagged_sent in tagged_sentences for word, tag in tagged_sent]))
    tags.sort()

    observation_dict = {word: i for i, word in enumerate(words)}
    state_dict = {state: i for i, state in enumerate(tags)}

    sentence_words = [[word for word, tag in tagged_sent] for tagged_sent in tagged_sentences]
    sentence_tags = [[tag for word, tag in tagged_sent] for tagged_sent in tagged_sentences]

    return words, tags, observation_dict, state_dict, \
           get_observation_ids(sentence_words, observation_dict), \
           get_state_ids(sentence_tags, state_dict)


def get_hindi_dataset():
    """
    generate the inputs required to train an hmm pos tagger for hindi dataset
    """
    all_tagged_sentences = indian.tagged_sents('hindi.pos')
    return generate_hmm_data(all_tagged_sentences)


class HMM:
    def __init__(self, num_states, num_observations):
        self.n = num_states
        self.m = num_observations

        # the small number added is to avoid log(0) error!
        self.pi = np.zeros(num_states) + 0.0000000001
        self.A = np.zeros((num_states, num_states)) + 0.0000000001
        self.B = np.zeros((num_states, num_observations)) + 0.0000000001

    def fit(self, state_ids: List[List[int]], observation_ids: List[List[int]]):
        """
        There is a one-to-one mapping between each element of state_ids and observations_ids
        
        state_ids: The list of list of tag ids of the tokens of the sentences
        observations_ids: The list of list of word ids of the tokens of the sentences
        
        ENTER CODE HERE: complete the code
        populate the parameters (self.pi, self.A, self.B) by counting the bi-grams
        for self.A use bi-grams of states and states
        for self.B use bi-grams of states and observations
        """
        for s_ids, o_ids in zip(state_ids, observation_ids):
            # count initial states
            
            self.pi[s_ids[0]] += 1
            
            # count state->state transitions
            for state_1, state_2 in zip(s_ids[:-1], s_ids[1:]):
                self.A[state_1, state_2] += 1
            
            
            # count state->observations emissions
            
            for state, observation in zip(s_ids, o_ids):
                self.B[state, observation] += 1
            
            # HINT: use zip for creating bi-grams

        # normalize the rows of each probability matrix
        self.pi = np.log(self.pi / np.sum(self.pi))
        self.A = np.log(self.A / np.sum(self.A, axis=1).reshape((-1, 1)))
        self.B = np.log(self.B / np.sum(self.B, axis=1).reshape((-1, 1)))

    def path_log_prob(self, state_ids: List[List[int]], observation_ids: List[List[int]]) -> np.array:
        """
        There is a one-to-one mapping between each element of state_ids and observations_ids
        
        state_ids: The list of list of tag ids of the tokens of the sentences
        observations_ids: The list of list of word ids of the tokens of the sentences
        
        A debugging helper function to calculate the path probability of a given sequence of states and observations
        """
        all_path_log_probs = []
        for sent_states, sent_observations in zip(state_ids, observation_ids):
            # initial prob and all transition probs
            transition_log_probs = np.array([self.pi[sent_states[0]]] +
                                            [self.A[t_1, t_2]
                                             for t_1, t_2 in zip(sent_states[:-1], sent_states[1:])])

            observation_log_probs = np.array([self.B[t, e] for t, e in zip(sent_states, sent_observations)])

            all_path_log_probs.append(transition_log_probs.sum() + observation_log_probs.sum())

        return np.array(all_path_log_probs)

    def decode(self, observation_ids: List[List[int]]) -> List[List[int]]:
        """
        ENTER CODE HERE: complete the code
        Viterbi Algorithm: Follow the algorithm in Jim's book:
        Figure 8.10 at https://web.stanford.edu/~jurafsky/slp3/8.pdf
        """
        # store the decoded states here
        all_predictions = []
        for obs_ids in observation_ids:
            T = len(obs_ids)  # Sequence length
            viterbi = np.zeros((self.n, T))  # The viterbi table
            back_pointer = np.zeros((self.n, T))   # backpointers for each state+sequence id
            print("viterbit: ", viterbi.shape)
            print("back pointer:", back_pointer.shape)
            # TODO: Fill the viterbi table, back_pointer. Get the optimal sequence by backtracking
            for s in range(self.n):
                viterbi[s][0] = self.pi[s] * self.B[s][obs_ids[0]]
                back_pointer[s][0] = 0
            
            for t in range(1, T):
                for s in range(self.n):
                    max_prob = 0
                    max_state = 0
                    for i in range(self.n):
                        prob = viterbi[i][t-1] * self.A[i][s] * self.B[s][obs_ids[t]]
                        if prob > max_prob:
                            max_prob = prob
                            max_state = i
                    viterbi[s][t] = max_prob
                    back_pointer[s][t] = max_state
            
            best_path_prob = [0] + T
            best_path_pointer = 
                
            
        
            ...
            # raise NotImplementedError
        return all_predictions


def test_fit():
    # Assume a set of observations: {0, 1, 2} and set of states {0, 1}
    test_n = 2
    test_m = 3

    test_hmm_fit = HMM(test_n, test_m)

    train_states = [[0, 1, 1], [1, 0, 1], [0, 0, 1]]
    train_obs = [[1, 0, 2], [0, 0, 1], [2, 1, 0]]

    test_hmm_fit.fit(train_states, train_obs)

    assert np.round(np.exp(test_hmm_fit.pi)[0], 3) == 0.667

    assert np.round(np.exp(test_hmm_fit.A)[0, 1], 2) == 0.75

    assert np.round(np.exp(test_hmm_fit.B)[1, 0], 1) == 0.6
    
    ## Smarter test case 1

    observations1 = [0]*3 + [1]*2 + [3]*1 + [4]*1 + [5]*2

    observations = observations1 * 4

    train_states = [0]*len(observations1) + [1]*len(observations1) + [2]*len(observations1) + [3]*len(observations1)

    test_hmm_2 = HMM(4, 6)

    test_hmm_2.fit([train_states], [observations])

    assert np.round(np.exp(test_hmm_2.B[0, 1]), 2) == 0.22
    assert np.round(np.exp(test_hmm_2.B[0, 0]), 2) == 0.33
    assert np.round(np.exp(test_hmm_2.B[0, -1]), 2) == 0.22

    assert np.sum(test_hmm_2.B[0] == test_hmm_2.B[1]) == 6
    assert np.sum(test_hmm_2.B[1] == test_hmm_2.B[2]) == 6

    ## Smarter test case 2

    train_states = [0, 1, 1, 2, 4, 3, 2, 1, 3, 4] * 4
    observations = [1, 0, 1, 1, 1, 4, 2, 3, 4, 1] * 4

    test_hmm_3 = HMM(5, 5)
    test_hmm_3.fit([train_states], [observations])

    assert np.round(np.exp(test_hmm_3.A), 2)[4, 0] == 0.43
    assert np.round(np.exp(test_hmm_3.A), 2)[2, 4] == 0.5
    
    print('All Test Cases Passed!')


def test_decode():
    # Assume a set of observations: {0, 1, 2} and set of states {0, 1}
    test_n = 2
    test_m = 3

    test_hmm_tagger = HMM(test_n, test_m)
    test_hmm_tagger.pi = np.log([0.5, 0.5])
    test_hmm_tagger.A = np.log([[0.3, 0.7], [0.6, 0.4]])
    test_hmm_tagger.B = np.log([[0.2, 0.5, 0.3], [0.3, 0.1, 0.6]])

    test_state_observation_ids = [[[1, 0, 1], [2, 1, 1]]]
    test_state_ids = [[1, 0, 1]]
    test_obs_ids = [[2, 1, 1]]
    test_forwards = test_hmm_tagger.path_log_prob(test_state_ids, test_obs_ids)

    expected_prob = 0.5 * 0.6 * 0.6 * 0.5 * 0.7 * 0.1

    predicted_forward = np.exp(test_forwards)[0]

    assert np.round(predicted_forward, 4) == expected_prob

    ### TEST Viterbi Decoding Method ###
    import itertools
    
    def brute_force(test_obs, n):
        T = len(test_obs)
        state_s = list(range(n))
        all_possibs = list(itertools.product(state_s, repeat=T))
        all_obs = [test_obs] * len(all_possibs)
        all_forwards = test_hmm_tagger.path_log_prob(all_possibs, all_obs)
        best_index = np.argmax(all_forwards)
        best_path_true = all_possibs[best_index]
        # print(best_path_true)
        return best_path_true

    test_observations = [2, 1, 1]

    bp_true = brute_force(test_observations, 2)

    decoded_states = test_hmm_tagger.decode([test_observations])
    
    assert tuple(decoded_states[0]) == bp_true

    test_observations = [1, 1, 1]

    bp_true = brute_force(test_observations, 2)

    decoded_states = test_hmm_tagger.decode([test_observations])

    assert tuple(decoded_states[0]) == bp_true

    test_observations = [1, 1, 2]

    bp_true = brute_force(test_observations, 2)

    decoded_states = test_hmm_tagger.decode([test_observations])

    assert tuple(decoded_states[0]) == bp_true

    print('All Test Cases Passed!')


def run_tests():
    print('Testing the fit function of the HMM')
    test_fit()

    print('Testing the decode function of the HMM')
    test_decode()

    print('Yay! You have a working HMM. Now try creating a pos-tagger using this class.')


if __name__ == "__main__":
    run_tests()
