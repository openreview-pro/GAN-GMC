import torch
import torch.nn as nn
# from torch.autograd import Variable


class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, num_filters, output_dim):
        super(Discriminator, self).__init__()

        # main layers
        self.main_layer = torch.nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=500),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 32, kernel_size=500),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 32, kernel_size=500),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 32, kernel_size=100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 1, kernel_size=86),
            nn.Sigmoid()
        )

    def forward(self, input):
        out = self.main_layer(input)
        return out

