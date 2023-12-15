import torch 
import numpy as np
from scipy import stats
from scipy.special import softmax

class SSM:
    def __init__(self, input_d, output_d):
        self.array_phi=torch.from_numpy(np.random.dirichlet(alpha=[1]*output_d, size=1)[0])

    def forward(self, z):
        return self.array_phi

        import numpy as np

    def phi(z_t):
        '''Computes the MAP estimate of the Dirichlet distribution'''

        numerator = n_wk + beta
        denominator = n_k + V @ beta
        return numerator / denominator

    def draw_latent_state(s_t):
        '''Samples from p(z_t; g(s_t))'''

        probas = stats.multinomial(n=vocabulary_length, compute_theta(s_t))
        return np.argmax(probas)

    def compute_theta(s_t):
        #W= #matrix that maps LSTM states to latent states
        #b= #bias term
        return softmax(W @ s_t + b)

    def draw_observation(z_t):
        '''Samples from p(x_t; h(z_t))'''

        probas = stats.multinomial(x_t, phi(z_t))
        return np.argmax(probas)

    def generative_process(sequence, topics):
        '''Generative process of SSL for a single sequence'''

        for t in range(1,T+1):
            #1. Perform LSTM transition
            s_t = LSTM(s_t, z_t)

            #2. Draw latent state
            z_t = draw_latent_state(s_t)

            #3. Draw observation
            x_t = draw_observation(z_t)

alpha_t = np.dot(theta_t, phi(x_t))
gamma_t = theta_t * phi(x_t)