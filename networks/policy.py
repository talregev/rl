import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):

    def __init__(self, device, k, num_states, num_actions) -> None:
        super(Policy, self).__init__()
        len_s = len(F.one_hot(torch.tensor(0), num_classes=num_states))
        self.fc1 = nn.Linear(len_s, k, device=device)
        self.fc2 = nn.Linear(k, num_actions, device=device)
        self.num_states = num_states
        self.num_actions = num_actions
        self.device = device

    def forward(self, s):
        input = F.one_hot(torch.tensor(s, device=self.device), num_classes=self.num_states).float()
        x1 = self.fc1(input)
        # Use the rectified-linear activation function over x
        x2 = F.relu(x1)
        x3 = self.fc2(x2)
        output = F.softmax(x3, dim=-1)
        return output
