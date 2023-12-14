import torch 
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_d, hidden_d, layer_d, output_d):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_d
        self.layer_dim = layer_d

        # LSTM model 
        self.lstm = nn.LSTM(input_d, hidden_d, layer_d) 
        self.fc = nn.Softmax(hidden_d, output_d)

    def forward(self, z):
    
        h0 = torch.zeros(self.layer_dim, z.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, z.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(z, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :]) 
        return out