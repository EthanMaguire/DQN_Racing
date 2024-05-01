import torch


# TODO Try using RNN, recurrent layers which have some memory
class DQN_1(torch.nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN_1, self).__init__()

        self.linear1 = torch.nn.Linear(n_observations, 60)  # Input layer?
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(60, 60)  # Hidden Layer?
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(60, 60)  # Hidden Layer?
        self.activation3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(60, n_actions)  # Output Layer

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        x = self.linear4(x)

        return x

