import torch
import numpy as np
from mixture_model import GMMMarginal

def viterbi_decode(log_likelihoods, l):
    T, num_states = log_likelihoods.shape
    path_cost = torch.full((T+1, num_states), float('inf'))
    path_cost[0] = 0
    backpointer = torch.zeros((T, num_states), dtype=torch.long)

    for t in range(1, T + 1):
        for s in range(num_states):
            prev_states = get_valid_prev_states(s, l)
            candidates = path_cost[t-1][prev_states] + (log_likelihoods[t-1][s])
            best_idx = torch.argmax(candidates)
            path_cost[t][s] = candidates[best_idx]
            backpointer[t-1][s] = prev_states[best_idx]

    best_path = torch.zeros(T, dtype=torch.long)
    best_path[T-1] = torch.argmin(path_cost[T])

    for t in range(T - 2, -1, -1):
        best_path[t] = backpointer[t][best_path[t+1]]
    return best_path


def get_valid_prev_states(curr_state, l):
    valid = []
    num_states = 2 ** l
    curr_bits = f"{curr_state:0{l}b}"

    for prev_state in range(num_states):
        prev_bits = f"{prev_state:0{l}b}"
        if prev_bits[1:] == curr_bits[:-1]:
            valid.append(prev_state)

    return torch.tensor(valid, dtype=torch.long)

def decode_with_viterbinet(model, Y_seq, X_seq, gmm_model, l, symbol_map={0: -1, 1: 1}):

    model.eval()
    with torch.no_grad():
        log_p_s_given_y = model(Y_seq)

        log_p_y = gmm_model.evaluate(Y_seq)

        log_p_y_given_s = compute_log_likelihoods(log_p_s_given_y, log_p_y, l)
        print("log_p_y_given_s: ", log_p_y_given_s.shape, log_p_y_given_s[0])

    decoded_states = viterbi_decode(log_p_y_given_s, l)
    print("States: ", decoded_states[:20])

    decoded_bits = []
    for idx in decoded_states:
        bits = [int(b) for b in f"{idx.item():0{l}b}"]
        decoded_bits.append(bits[0])

    decoded_bits = decoded_bits[:-(l-1)] if l > 1 else decoded_bits

    decoded_symbols = [symbol_map[b] for b in decoded_bits]

    return decoded_symbols, decoded_states[1:]

def compute_log_likelihoods(p_s_given_y_log, log_p_y, l):
    """
    Assume uniform p(s)
    p_s_given_y_log: (T, num_states), log p(s | y[i]), softmax output
    log_p_y: from GMM
    l: memory length
    """

    # num_states = p_s_given_y_log.shape[1]
    # log_p_s= -torch.log(torch.tensor(float(num_states)))
    # log_joint = p_s_given_y_log + log_p_y.unsqueeze(1)
    # log_p_y_given_s = log_joint - log_p_s

    log_p_y_expanded = log_p_y.unsqueeze(1).expand_as(p_s_given_y_log)

    log_p_y_given_s = p_s_given_y_log + log_p_y_expanded - np.log(1/8)

    return log_p_y_given_s
