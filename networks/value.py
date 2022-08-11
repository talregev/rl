import torch
import torch.nn as nn
import torch.nn.functional as F


class Value(nn.Module):

    def __init__(self, k, num_states) -> None:
        super(Value, self).__init__()
        len_s = len(F.one_hot(torch.tensor(0), num_classes=num_states))
        self.fc1 = nn.Linear(len_s, k)
        self.fc2 = nn.Linear(k, 1)
        self.num_states = num_states

    def forward(self, s):
        input_s = F.one_hot(torch.tensor(s), num_classes=self.num_states).float()
        x = self.fc1(input_s)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        x = self.fc2(x)
        output = F.relu(x)
        return output
