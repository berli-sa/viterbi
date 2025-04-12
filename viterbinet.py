import torch
import torch.nn as nn
import torch.nn.functional as F

class ViterbiNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_states, window_size=1):
        super(ViterbiNet, self).__init__()
        self.fc1 = nn.Linear(window_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_states)
        self.dropout = nn.Dropout(0.3)
        self.window_size = window_size
    
    def forward(self, y):

        if len(y.shape) == 3:
            batch_size, seq_len, _ = y.shape
            y = y.reshape(batch_size, -1)
        elif len(y.shape) == 2 and y.shape[1] == 1 and self.window_size > 1:
            y = y.repeat(1, self.window_size)

        x = F.relu(self.bn1(self.fc1(y)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
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