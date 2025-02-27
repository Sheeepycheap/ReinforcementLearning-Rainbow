import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    """
    A linear (fully connected) layer that adds learned noise to its weights and biases,
    based on 'Noisy Networks for Exploration' (Fortunato et al., 2018).
    
    The idea:
      weight = weight_mu + weight_sigma * epsilon
      bias   = bias_mu   + bias_sigma   * epsilon
    
    Both (weight_mu, weight_sigma) and (bias_mu, bias_sigma) are learnable parameters.
    The random noise epsilon is re-sampled each time we call reset_noise().
    
    By doing so, the network's outputs become stochastic in a learned manner, 
    and we don't need a separate epsilon-greedy exploration schedule.
    """

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        std_init: float = 0.5,
    ):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        # std_init is the initial scale for weight_sigma
        self.std_init = std_init

        # Mean parameters for weights/bias (learnable)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))

        # Sigma parameters for weights/bias (learnable)
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        # Tensors for storing sampled noise (NOT learnable, but re-sampled each update)
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))
        self.register_buffer("bias_epsilon",   torch.Tensor(out_features))

        # Initialize everything
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        Initialize the parameters.
        We pick weight_mu in [-mu_range, mu_range], 
        and set weight_sigma to a default scale ~ std_init / sqrt(in_features).
        
        This uses factorized noise logic to keep things stable.
        """
        mu_range = 1.0 / math.sqrt(self.in_features)

        # Set weight_mu to uniform in [-mu_range, mu_range]
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        # Set weight_sigma to a constant ~ std_init / sqrt(in_features)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        # Do the same for bias mu/sigma
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """
        Resample random noise for weight_epsilon and bias_epsilon.
        We'll do this typically once per training step, 
        so each forward pass in that step uses the same epsilon.
        """
        epsilon_in  = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # Outer product for weight noise 
        #  (factorized noise = epsilon_out x epsilon_in)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        # Bias noise just uses epsilon_out
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """
        Factorized noise approach:
          1) sample from normal distribution
          2) apply sign(x)*sqrt(|x|)
        This helps stabilize how noise scales with dimension.
        """
        x = torch.randn(size)  # draw from normal(0,1)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass uses:
          weight = weight_mu + weight_sigma * weight_epsilon
          bias   = bias_mu   + bias_sigma   * bias_epsilon
        Then does a standard linear transform with those noisy parameters.
        """
        # Compute final weight and bias
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon

        return F.linear(x, weight, bias)


class Network(nn.Module):
    """
    A DQN-like network that:
      1) Uses a common feature extractor (MLP with 128 hidden units).
      2) Splits into a dueling architecture:
          - advantage stream (NoisyLinear layers)
          - value stream (NoisyLinear layers)
      3) Outputs a distribution over atoms (C51),
         so we have 'atom_size' possible returns for each action.
      4) Summarizes that distribution in forward() by computing Q-values (expected returns).
    """

    def __init__(
        self, 
        in_dim: int,    # input dimension (e.g. state dimension)
        out_dim: int,   # number of actions
        atom_size: int, # number of discrete return-atom bins
        support: torch.Tensor  # the actual values of these bins [v_min..v_max]
    ):
        super(Network, self).__init__()
        
        self.support = support       # tensor of shape (atom_size,)
        self.out_dim = out_dim       # number of possible actions
        self.atom_size = atom_size   # number of atoms

        # Shared feature extractor: e.g. a simple MLP with 128 hidden units
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
        )
        
        # Advantage stream (Dueling)
        # We'll have 2 NoisyLinear layers:
        #  1) a hidden layer
        #  2) output layer that eventually yields advantage for each (action, atom)
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer        = NoisyLinear(128, out_dim * atom_size)

        # Value stream (Dueling)
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer        = NoisyLinear(128, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass to get Q-values. 
        We first get the distribution over returns for each action (dist(x)), 
        then do an expectation over the atoms.
        
        The final shape is (batch_size, out_dim), i.e. one Q-value per action.
        """
        dist = self.dist(x)    # shape: [batch_size, out_dim, atom_size]
        # Weighted sum over the distribution bins => Q(s, a) = sum_z( z * p(z) )
        q = torch.sum(dist * self.support, dim=2)  
        # shape: [batch_size, out_dim]
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """
        This computes the distribution over returns for each action. 
        Because it's dueling:
          - we get advantage logits (batch_size, out_dim * atom_size)
          - we get value logits     (batch_size, 1 * atom_size)
          - reshape them, combine them, then do a softmax over the 'atom' dimension.
        """
        feature = self.feature_layer(x)  # [batch_size, 128]

        # Advantage head
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        advantage = self.advantage_layer(adv_hid)
        advantage = advantage.view(-1, self.out_dim, self.atom_size) 
          # shape: [batch_size, out_dim, atom_size]

        # Value head
        val_hid = F.relu(self.value_hidden_layer(feature))
        value = self.value_layer(val_hid)
        value = value.view(-1, 1, self.atom_size)
          # shape: [batch_size, 1, atom_size]

        # Combine them in the dueling fashion:
        # q_atoms[a] = value + advantage[a] - mean(advantage[a_over_all])
        # Then apply softmax along the atom dimension:
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)   # probability over the atom dimension
        dist = dist.clamp(min=1e-3)         # small clamp to avoid numerical issues

        return dist  # shape: [batch_size, out_dim, atom_size]
    
    def reset_noise(self):
        """
        Reset the noise parameters in all NoisyLinear layers. 
        Typically called once per training iteration so the network 
        uses new noise samples for the next forward pass.
        """
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()