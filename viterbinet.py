import torch
import torch.nn as nn
import torch.nn.functional as F

class ViterbiNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_states):
        super(ViterbiNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_states)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, y):
        x = F.relu(self.bn1(self.fc1(y)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)