import torch
import torch.nn as nn
import torch.nn.functional as F


class QValue(nn.Module):

    def __init__(self, device, k, num_states, num_actions) -> None:
        super(QValue, self).__init__()
        len_s = len(F.one_hot(torch.tensor(0), num_classes=num_states))
        len_a = len(F.one_hot(torch.tensor(0), num_classes=num_actions))
        self.fc1 = nn.Linear(len_s + len_a, k, device=device)
        self.fc2 = nn.Linear(k, 1, device=device)
        self.num_states = num_states
        self.num_actions = num_actions
        self.device = device

    def forward(self, s, a):
        input_s = F.one_hot(torch.tensor(s, device=self.device), num_classes=self.num_states).float()
        input_a = F.one_hot(torch.tensor(a, device=self.device), num_classes=self.num_actions).float()
        x = torch.cat((input_s, input_a), -1)
        x = self.fc1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        x = self.fc2(x)
        output = F.relu(x)
        return output
