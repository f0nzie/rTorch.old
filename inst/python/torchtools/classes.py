import torch

class ConvNet(torch.nn.Module):
    def __init__(self, output_dim):
        super(ConvNet, self).__init__()

