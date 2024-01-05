import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model_length, batch_size, device="cpu"):
        """
        LSTM generative model.

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden state in the LSTM.
            output_size (int): Size of the output features.
            model_length (int): Length of the input and target sequences in the dataset.
            batch_size (int): Number of sequences in each batch.
            device ("str"): name of the device used. Default to "cpu".
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model_length = model_length
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.device = device

    def forward(self, x, prev_state):
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input sequence tensor.
            prev_state (tuple): Tuple containing the previous hidden and cell states.

        Returns:
            tuple: Tuple containing the output tensor and the new hidden and cell states.
        """
        x = x.to(self.device)
        output, state = self.lstm(x, prev_state)
        output = self.dropout(output)
        output = self.fc(output)
        return output, state

    def init_state(self):
        """
        Initialize the hidden and cell states.

        Returns:
            tuple: Tuple containing the initial hidden and cell states.
        """
        return (torch.zeros(1, self.batch_size, self.hidden_size).to(self.device),
                torch.zeros(1, self.batch_size, self.hidden_size).to(self.device))

    def train_model(self, dataset, optimizer, criterion):
        """
        Train the LSTM model using the provided dataset.

        Args:
            dataset (torch.utils.data.Dataloader): PyTorch dataloader for training.
            optimizer (torch.optim.Optimizer): Optimization algorithm.
            criterion (torch.nn.Module): Loss function.
        """
        self.train()
        state_h, state_c = self.init_state()
        lst_output = []
        lst_y_true = []
        optimizer.zero_grad()

        for t, (x, y) in enumerate(dataset):
            x = x.to(self.device)
            y = y.to(self.device)
            output, (state_h, state_c) = self.forward(x, (state_h, state_c))
            state_h = state_h.detach()
            state_c = state_c.detach()
            lst_output.append(output[:, -1, :])
            lst_y_true.append(y[:, -1, :])
        torch_output = torch.stack(lst_output).view(-1, self.output_size)
        torch_y_true = torch.stack(lst_y_true).view(-1, self.output_size)
        loss = criterion(torch_output, torch_y_true)
        loss.backward()
        optimizer.step()

    def predict_next_probability(self, input_sequence):
        """
        Predict the next probability distribution given an input sequence.

        Args:
            input_sequence (torch.Tensor): Input sequence, one hot encoded tensor.

        Returns:
            torch.Tensor: Predicted probability distribution.
        """
        self.eval()
        input_sequence = input_sequence.to(self.device)
        state_h, state_c = self.init_state()
        with torch.no_grad():
            for t in range(len(input_sequence)):
                input_t = input_sequence[t].unsqueeze(0).unsqueeze(0)
                _, (state_h, state_c) = self(input_t, (state_h, state_c))

        input_t = input_sequence[-1].unsqueeze(0).unsqueeze(0)
        output, _ = self.forward(input_t, (state_h, state_c))
        probabilities = F.softmax(output[:, -1, :], dim=1).detach().squeeze(0)
        return probabilities

    def sample_next_z(self, input_sequence):
        """
        Sample the next vocabulary index given an input sequence using multinomial sampling.

        Args:
            input_sequence (torch.Tensor): Input sequence, one hot encoded tensor.

        Returns:
            int: Sampled value.
        """
        self.eval()
        proba = self.predict_next_probability(input_sequence)
        return torch.multinomial(proba, 1).item() + 1
