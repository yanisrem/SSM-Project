import torch
import torch.nn.functional as F

def compute_alpha_unnormalized(z_1_t_minus_1, num_topics, num_voc, lstm, ssm):
    """
    Compute unnormalized alpha values for the current time step.

    Args:
        z_1_t_minus_1 (torch.Tensor): Previous topic assignments from time 1 to t-1.
        num_topics (int): Number of topics.
        num_voc (int): Size of the vocabulary.
        lstm (LSTM): LSTM model for topic generation.
        ssm (SSM): State Space model for text generation.

    Returns:
        torch.Tensor: Unnormalized alpha values.
    """
    z_1_t_minus_1 = z_1_t_minus_1 - 1
    z_one_hot = F.one_hot(z_1_t_minus_1, num_classes=num_topics).float()
    softmax = lstm.predict_next_probability(z_one_hot).detach()
    softmax = softmax.cpu()
    phi = ssm.phi
    alpha = torch.tensor([torch.matmul(softmax, phi[j, :]) for j in range(num_voc)])
    return alpha

def compute_alpha_normalized(z_1_t_minus_1, num_topics, num_voc, lstm, ssm):
    """
    Compute normalized alpha values for the current time step.

    Args:
        z_1_t_minus_1 (torch.Tensor): Previous topic assignments from time 1 to t-1.
        num_topics (int): Number of topics.
        num_voc (int): Size of the vocabulary.
        lstm (LSTM): LSTM model for topic generation.
        ssm (SSM): State Space model for text generation.

    Returns:
        torch.Tensor: Normalized alpha values.
    """
    num = compute_alpha_unnormalized(z_1_t_minus_1, num_topics, num_voc, lstm, ssm)
    denom = torch.sum(num) + 1e-6
    return num / denom

def compute_gamma_unnormalized(xt, z_1_t_minus_1, num_topics, lstm, ssm):
    """
    Compute unormalized gamma values for the current time step.

    Args:
        xt (torch.Tensor) : Encoded text sample at time t
        z_1_t_minus_1 (torch.Tensor): Previous topic assignments from time 1 to t-1.
        num_topics (int): Number of topics.
        lstm (LSTM): LSTM model for topic generation.
        ssm (SSM): State Space model for text generation.

    Returns:
        torch.Tensor: Unormalized gamma values.
    """
    z_1_t_minus_1 = z_1_t_minus_1 - 1
    z_one_hot = F.one_hot(z_1_t_minus_1, num_classes=num_topics).float()
    softmax = lstm.predict_next_probability(z_one_hot).detach()
    softmax = softmax.cpu()
    phi = ssm.phi
    phi_xt = phi[xt - 1, :]
    return torch.mul(softmax, phi_xt)

def compute_gamma_normalized(xt, z_1_t_minus_1, num_topics, lstm, ssm):
    """
    Compute normalized gamma values for the current time step.

    Args:
        xt (torch.Tensor) : Encoded text sample at time t
        z_1_t_minus_1 (torch.Tensor): Previous topic assignments from time 1 to t-1.
        num_topics (int): Number of topics.
        lstm (LSTM): LSTM model for topic generation.
        ssm (SSM): State Space model for text generation.

    Returns:
        torch.Tensor: Normalized gamma values.
    """
    num = compute_gamma_unnormalized(xt, z_1_t_minus_1, num_topics, lstm, ssm)
    denom = torch.sum(num) + 1e-6
    return num / denom

def particle_gibbs(x, previous_z_1_T_star, P, num_topics, num_words, T, lstm_model, ssm_model):
    """
    Sample an estimation of the unobserved topics.

    Args:
        x (torch.Tensor) : Encoded text sample
        previous_z_1_T_star (torch.Tensor): Topics sampled at previous iteration.
        P (int): Number of particles
        num_topics (int): Number of topics.
        num_words (int): Size of the vocabulary.
        T (int) sequence length
        lstm_model (LSTM): LSTM model for topic generation.
        ssm_model (SSM): State Space model for text generation.

    Returns:
        torch.Tensor: Sampled topics
    """

    # Init
    Z_matrix = torch.zeros((P, T + 1), dtype=torch.long)
    alpha_matrix = torch.zeros((P, T + 1), dtype=torch.float)
    ancestor_matrix = torch.ones((P, T + 1), dtype=torch.long)

    # t=0
    Z_matrix[:, 0] = torch.randint(1, num_topics + 1, (P,))
    alpha_matrix[:, 0] = torch.full((P,), 1 / P)

    # t=1
    t = 1
    ancestor_matrix[0, t - 1] = torch.tensor(1)
    Z_matrix[0, 1:t + 1] = previous_z_1_T_star[:t]

    for p in range(2, P + 1):
        alpha_t_minus_1_p = alpha_matrix[:, t - 1]
        a_t_minus_1_p = torch.multinomial(alpha_t_minus_1_p, 1).item()+1
        ancestor_matrix[p - 1, t - 1] = a_t_minus_1_p
        z_1_t_minus_1_a_t_minus_1_p = Z_matrix[int(a_t_minus_1_p) - 1, 0]
        z_1_t_minus_1_a_t_minus_1_p = torch.tensor([z_1_t_minus_1_a_t_minus_1_p])
        gamma_t_p = compute_gamma_normalized(xt=x[t - 1],
                                             z_1_t_minus_1=z_1_t_minus_1_a_t_minus_1_p,
                                             num_topics=num_topics,
                                             lstm=lstm_model,
                                             ssm=ssm_model)
        Z_matrix[p - 1, 1:t + 1] = torch.multinomial(gamma_t_p, 1).item()+1

    for p in range(1, P + 1):
        a_t_minus_1_p = ancestor_matrix[p - 1, t - 1]
        z_1_t_minus_1_a_t_minus_1_p = Z_matrix[int(a_t_minus_1_p) - 1, 0]
        z_1_t_minus_1_a_t_minus_1_p = torch.tensor([z_1_t_minus_1_a_t_minus_1_p])
        alpha_t_p = compute_alpha_normalized(z_1_t_minus_1=z_1_t_minus_1_a_t_minus_1_p,
                                             num_topics=num_topics,
                                             num_voc=num_words,
                                             lstm=lstm_model,
                                             ssm=ssm_model)
        alpha_t_p = alpha_t_p[x[t - 1] - 1]
        alpha_matrix[p - 1, t] = alpha_t_p

    alpha_matrix[:, t] = alpha_matrix[:, t] / (alpha_matrix[:, t].sum() + 1e-6)

    for t in range(2, T + 1):
        a_t_minus_1 = torch.tensor(1)
        z_1_t = previous_z_1_T_star[:t]
        ancestor_matrix[0, t - 1] = a_t_minus_1
        Z_matrix[0, 1:t + 1] = z_1_t

        for p in range(2, P + 1):
            alpha_t_minus_1_p = alpha_matrix[:, t - 1]
            a_t_minus_1_p = torch.multinomial(alpha_t_minus_1_p, 1).item()+1
            ancestor_matrix[p - 1, t - 1] = a_t_minus_1_p
            z_1_t_minus_1_a_t_minus_1_p = Z_matrix[int(a_t_minus_1_p) - 1, 1:t]
            gamma_t_p = compute_gamma_normalized(xt=x[t - 1],
                                                 z_1_t_minus_1=z_1_t_minus_1_a_t_minus_1_p,
                                                 num_topics=num_topics,
                                                 lstm=lstm_model,
                                                 ssm=ssm_model)

            z_t_p = torch.multinomial(gamma_t_p, 1).item()+1
            z_1_t_p = torch.cat([z_1_t_minus_1_a_t_minus_1_p, torch.tensor([z_t_p])])
            Z_matrix[p - 1, 1:t + 1] = z_1_t_p

        for p in range(1, P + 1):
            a_t_minus_1_p = ancestor_matrix[p - 1, t - 1]
            z_1_t_minus_1_a_t_minus_1_p = Z_matrix[int(a_t_minus_1_p) - 1, 1:(t - 1) + 1]
            alpha_t_p = compute_alpha_normalized(z_1_t_minus_1=z_1_t_minus_1_a_t_minus_1_p,
                                                 num_topics=num_topics,
                                                 num_voc=num_words,
                                                 lstm=lstm_model,
                                                 ssm=ssm_model)
            alpha_t_p = alpha_t_p[x[t - 1] - 1]
            alpha_matrix[p - 1, t] = alpha_t_p

        alpha_matrix[:, t] = alpha_matrix[:, t] / (alpha_matrix[:, t].sum() + 1e-6)

    alpha_T = alpha_matrix[:, -1]
    alpha_T = alpha_T / (alpha_T.sum() + 1e-6)
    r = torch.multinomial(alpha_T, 1).item()+1
    a_T_r = ancestor_matrix[int(r) - 1, -1]
    z_1_T = Z_matrix[int(a_T_r) - 1, 1:]

    return z_1_T