from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np
import torch

class KDEMarginal:
    def __init__(self, bandwidth=0.03, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
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
    
    def tune(self, Y_train, bandwidth=None, cv=5, verbose=True):
        if isinstance(Y_train, torch.Tensor):
            Y_train = Y_train.cpu().numpy()
        Y_train = Y_train.reshape(-1, 1)

        if bandwidth is None:
            bandwidths = np.logspace(-1.5, 1.0, 20)

        grid = GridSearchCV(KernelDensity(kernel=self.kernel), {'bandwidth': bandwidths}, cv=cv, verbose=3)
        grid.fit(Y_train)

        self.model = grid.best_estimator_
        self.bandwidth = grid.best_params_['bandwidth']

        print(f"[KDEMarginal] Best bandwidth selected: {self.bandwidth:.4f}")