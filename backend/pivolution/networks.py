import numpy as np


class Perceptron:
    def __init__(self, n_inputs, n_outputs, hidden_size):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_size = hidden_size

        self.weight_sizes = [
            n_inputs * hidden_size,     # layer 1 weights
            hidden_size,                # layer 1 bias
            hidden_size * n_outputs,    # layer 2 weights
            n_outputs                   # layer 2 bias
        ]

    def set_weights(self, params):
        assert len(params) == sum(self.weight_sizes)
        params = np.split(params, np.cumsum(self.weight_sizes))

        self.weight_1 = params[0].reshape(self.n_inputs, self.hidden_size)
        self.weight_1 = self.weight_1 / np.sqrt(self.n_inputs)
        self.bias_1 = params[1][None] / 10

        self.weights_2 = params[2].reshape(self.hidden_size, self.n_outputs)
        self.weights_2 = self.weights_2 / np.sqrt(self.hidden_size)
        self.bias_2 = params[3][None] / 10

    def forward(self, feats):
        # Layer 1
        hidden = feats[None] @ self.weight_1 + self.bias_1
        # ReLU
        hidden = np.clip(hidden, 0, None)
        # Layer 2
        out = hidden @ self.weights_2 + self.bias_2
        out = out.flatten()
        return out

    def get_new_params(self):
        return np.random.normal(size=sum(self.weight_sizes))


class Recurrent:
    def __init__(self, n_inputs, n_outputs, hidden_size, state_size):
        self.n_outputs = n_outputs
        self.net = Perceptron(
            n_inputs + state_size,
            n_outputs + 2 * state_size,
            hidden_size
        )
        self.state = np.zeros(state_size)
    
    def set_weights(self, params):
        self.net.set_weights(params)
    
    def forward(self, feats):
        feats = np.hstack([feats, self.state])
        out = self.net.forward(feats)

        state_update = out[self.n_outputs:]
        gate = np.clip(state_update[:len(self.state)] + 0.5, 0, 1)
        new_state = state_update[len(self.state):]
        self.state = self.state * gate + new_state * (1 - gate)

        return out[:self.n_outputs]

    def get_new_params(self):
        return self.net.get_new_params()