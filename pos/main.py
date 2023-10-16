import numpy as np


def decode(observation_ids):
    """
    ENTER CODE HERE: complete the code
    Viterbi Algorithm: Follow the algorithm in Jim's book:
    Figure 8.10 at https://web.stanford.edu/~jurafsky/slp3/8.pdf
    """
    # store the decoded states here
    pi = np.array([0.8, 0.2])
    n = len(pi)
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.2, 0.4, 0.4], [0.5, 0.4, 0.1]])
    all_predictions = []
    for obs_ids in observation_ids:
        T = len(obs_ids)  # Sequence length
        viterbi = np.zeros((n, T))  # The viterbi table
        back_pointer = np.zeros((n, T))  # backpointers for each state+sequence id
        # TODO: Fill the viterbi table, back_pointer. Get the optimal sequence by backtracking
        for s in range(n):
            p = pi[s]
            bs = B[s]
            b = B[s][obs_ids[0]-1]
            obs_id = obs_ids[0]

            viterbi[s][0] = p * b
            back_pointer[s][0] = 0
        for t in range(1, T):
            # print(f"Viterbit{t}: {viterbi} ")
            # print(f"back_pointer{t}: {back_pointer}")

            for s in range(n):
                max_prob = 0
                max_state = 0
                for i in range(n):
                    v = viterbi[i][t - 1]
                    a = A[i][s]
                    b = B[s][obs_ids[t] - 1]
                    prob = v * a * b
                    if prob > max_prob:
                        max_prob = prob
                        max_state = i
                viterbi[s][t] = max_prob
                back_pointer[s][t] = max_state
        best_path = [0] * T
        print("v[T-1]:",viterbi[:, T - 1])
        print("argmax positon: ",np.argmax(viterbi[:, T - 1]))
        best_path[T - 1] = np.argmax(viterbi[:, T - 1])  # type: ignore
        print("v:\n", viterbi)
        print("back pointers:\n", back_pointer)
        print("best path:", best_path)


if __name__ == "__main__":
    test_state_observation_ids = np.array([[1, 3, 1], [1, 1, 1]])
    decode(test_state_observation_ids)
