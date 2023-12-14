import torch 
import numpy as np
class SSM:
    def __init__(self, input_d, output_d):
        self.array_phi=torch.from_numpy(np.random.dirichlet(alpha=[1]*output_d, size=1)[0])

    def forward(self, z):
        return self.array_phi