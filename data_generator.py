import numpy as np

def generate_isi_data(num_samples, l, snr_db):
    S = np.random.choice([-1, 1], size=(num_samples + l - 1))
    # S = np.ones(num_samples + l - 1)
    h = np.exp(-0.3 * np.arange(l))
    X = np.array([S[i:i+l] for i in range(num_samples)])
    noise_std = 10 ** (-snr_db / 20)
    Y = np.array([np.dot(X[i], h[::-1]) + np.random.normal(0, noise_std) for i in range(num_samples)])
    return Y, X, S