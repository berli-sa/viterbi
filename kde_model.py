from sklearn.neighbors import KernelDensity
import numpy as np
import torch

class KDEMarginal:
    def __init__(self, bandwidth=0.2, kernel='gaussian'):
        self.model = KernelDensity(bandwidth=bandwidth, kernel=kernel)

    def fit(self, Y_train):
        if isinstance(Y_train, torch.Tensor):
            Y_train = Y_train.cpu().numpy()
        Y_train = Y_train.reshape(-1, 1)
        self.model.fit(Y_train)
    
    def evaluate(self, Y_seq):
        if isinstance(Y_seq, torch.Tensor):
            Y_seq = Y_seq.cpu().numpy()
        Y_seq = Y_seq.reshape(-1, 1)
        log_p_y = self.model.score_samples(Y_seq)
        return torch.tensor(log_p_y, dtype=torch.float32)