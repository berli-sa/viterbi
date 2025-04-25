import torch
import torch.nn as nn
import torch.nn.functional as F

class ViterbiNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_states, window_size=1):
        super(ViterbiNet, self).__init__()
        layers = [nn.Linear(1, hidden_dim),
                  nn.ReLU(),
                  nn.Linear(hidden_dim, num_states)]
        self.net = nn.Sequential(*layers)
        self.window_size = window_size
    
    def forward(self, y):
        x = self.net(y)
        return F.log_softmax(x, dim=1)

def create_sliding_window(sequence, window_size=3):
    if len(sequence.shape) == 3:
        batch_size, seq_len, features = sequence.shape
        result = torch.zeros(batch_size, seq_len, window_size, device=sequence.device)

        for i in range(seq_len):
            for w in range(window_size):
                idx = max(0, min(seq_len - 1, i + w - window_size // 2))
                result[:, i, w] = sequence[:, idx, 0]
        
        return result
    else:
        seq_len, features = sequence.shape
        result = torch.zeros(seq_len, window_size, device=sequence.device)

        for i in range(seq_len):
            for w in range(window_size):
                idx = max(0, min(seq_len - 1, i + w - window_size // 2))
                result[i, w] = sequence[idx, 0]

        return result