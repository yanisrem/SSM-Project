import torch

class Dataset(torch.utils.data.Dataset):
    """Dataset class for custom PyTorch dataset.

    Args:
        topics (torch.tensor): Tensor of one-hot encoded topic vectors.
        model_length (int): Length of the input and target sequences in the dataset.
    """
    def __init__(self, topics, model_length):
        """Class constructor

        Args:
            topics (torch.tensor): Tensor of one-hot encoded topic vectors.
            model_length (int): Length of the input and target sequences in the dataset.
        """
        self.topics = topics
        self.model_length = model_length

    def __len__(self):
        """Returns the effective length of the dataset.

        Returns:
            int: Effective length of the dataset.
        """
        return len(self.topics) - self.model_length

    def __getitem__(self, index):
        """Returns input and target sequences for a given index.

        Args:
            index (int): Index of the dataset to retrieve.

        Returns:
            tuple: A tuple containing the input sequence and the target sequence.
        """
        input_sequence = torch.tensor(self.topics[index:index + self.model_length, :])
        target_sequence = torch.tensor(self.topics[index + 1:index + self.model_length + 1, :])

        return input_sequence, target_sequence
