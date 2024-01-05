import torch

class SSM:
    def __init__(self, num_words, num_topics, sequence_length):
        """
        State Space Model (SSM) class.

        Args:
            num_words (int): Size the vocabulary.
            num_topics (int): Number of topics.
            sequence_length (int): Length of the sequence.
        """
        self.num_words = num_words
        self.num_topics = num_topics
        self.sequence_length = sequence_length
        self.phi = torch.randn(num_words, num_topics) * 0.01
        self.phi = torch.exp(self.phi - torch.max(self.phi, dim=0, keepdim=True).values)
        self.phi = self.phi / torch.sum(self.phi, dim=0, keepdim=True)

    def compute_MLE_SSM(self, ech_x, ech_z):
        """
        Compute Maximum Likelihood Estimate (MLE) for the SSM parameters.

        Args:
            ech_x (torch.Tensor): Samples of text sequences.
            ech_z (torch.Tensor): Samples of topic sequences.
        """
        T = self.sequence_length
        proba_matrix = torch.zeros((self.num_words, self.num_topics), dtype=torch.float)
        for t in range(T):
            vect_x_t = ech_x[:, t]
            vect_z_t = ech_z[:, t]
            for i in range(len(vect_x_t)):
                proba_matrix[vect_x_t[i] - 1, vect_z_t[i] - 1] += 1
        proba_matrix = proba_matrix + 1e-6
        row_sums = proba_matrix.sum(dim=0, keepdim=True)
        proba_matrix_normalized = proba_matrix / row_sums
        self.phi = proba_matrix_normalized

    def predict_proba(self, z_t):
        """
        Predict the probability distribution over words given a topic assignment.

        Args:
            z_t (int): Topic assignment at time t.

        Returns:
            torch.Tensor: Probability distribution over words.
        """
        return self.phi[:, int(z_t) - 1]

    def sample_xt(self, z_t):
        """
        Sample a word given a topic assignment.

        Args:
            z_t (int): Topic assignment at time t.

        Returns:
            int: Sampled observation.
        """
        proba = self.predict_proba(z_t)
        sampled_xt = torch.multinomial(proba, 1).item() + 1
        return sampled_xt
