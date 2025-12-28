import numpy as np


class Brain:
    def __init__(self, input_size=7, hidden_size=12, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases with Gaussian distribution
        # W1: Input -> Hidden
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)

        # W2: Hidden -> Output
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.random.randn(output_size)

    def forward(self, inputs):
        """
        Performs forward pass through the network.
        inputs: numpy array of shape (input_size,) or (batch_size, input_size)
        """
        # Ensure inputs is at least 1D numpy array
        x = np.array(inputs)

        # Layer 1
        z1 = np.dot(x, self.w1) + self.b1
        a1 = np.tanh(z1)  # Tanh activation

        # Layer 2
        z2 = np.dot(a1, self.w2) + self.b2

        # Output (Softmax)
        # Numerical stability trick: subtract max before exp
        exp_z2 = np.exp(z2 - np.max(z2))
        output = exp_z2 / np.sum(exp_z2)

        return output

    def get_params(self):
        """Returns tuple of (w1, b1, w2, b2) for batch processing."""
        return self.w1, self.b1, self.w2, self.b2

    def mutate(self, rate=0.01, magnitude=0.1):
        """
        Applies Gaussian noise to weights and biases.
        rate: Probability of a weight being mutated (not used in simple Gaussian addition,
              but often used in sparse mutation. Here we'll stick to simple additive noise
              as implied by 'Gaussian Noise added to the weights').
        magnitude: Standard deviation of the noise.
        """
        # Add noise to all weights (simple evolution strategy)
        self.w1 += np.random.randn(*self.w1.shape) * magnitude
        self.b1 += np.random.randn(*self.b1.shape) * magnitude
        self.w2 += np.random.randn(*self.w2.shape) * magnitude
        self.b2 += np.random.randn(*self.b2.shape) * magnitude
