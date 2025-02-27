import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import Dict, Tuple, List
from torch.nn.utils import clip_grad_norm_
from Replay_buffer import PrioritizedReplayBuffer, ReplayBuffer 
from Networks import Network  

class DQNAgent:
    """
    A DQN-based agent that integrates multiple Rainbow components:
      - Prioritized Replay (PER)
      - Multi-step returns
      - Noisy Nets
      - Dueling network architecture
      - C51 Distributional RL
      - Double DQN 
    It interacts with the environment and optimizes a categorical DQN.
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        seed: int,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 3,
    ):
        """
        Args:
            env (gym.Env): The environment to interact with.
            memory_size (int): How many transitions the replay buffer can hold.
            batch_size (int): Number of samples per training batch.
            target_update (int): Interval (steps) at which we update the target network.
            seed (int): For reproducibility with environment resets.
            gamma (float): Discount factor for rewards.
            alpha (float): Priority exponent in Prioritized Replay.
            beta (float): Importance-sampling exponent in Prioritized Replay.
            prior_eps (float): Small constant added to TD-error to avoid zero priority.
            v_min (float): Minimum possible return for C51.
            v_max (float): Maximum possible return for C51.
            atom_size (int): Number of discrete bins for the return distribution.
            n_step (int): Number of steps to accumulate for multi-step returns.
        """

        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        self.seed = seed

        # We remove epsilon-greedy logic because NoisyNet does exploration internally
        # (i.e., no separate epsilon schedule is needed).

        # Device selection (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running on device:", self.device)
        
        # -----------------------------
        # Prioritized Replay (PER)
        # -----------------------------
        # Beta adjusts how heavily we apply importance sampling corrections.
        # We store transitions in a prioritized structure for 1-step transitions,
        # but also keep an n-step buffer if needed.
        self.beta = beta
        self.prior_eps = prior_eps

        # 1-step memory
        
        self.memory = PrioritizedReplayBuffer(
            obs_dim=env.observation_space.shape[0],
            size=memory_size,
            batch_size=batch_size,
            alpha=alpha,      # how strongly priority matters
            gamma=gamma,      # discount factor used in n-step logic
            n_step=1
        )
        
        # If we want multi-step returns, we also keep a separate n-step buffer,
        # used to generate transitions for the primary PER buffer.
        # The original Rainbow uses only multi-step, but here we combine 1-step + n-step.
        self.use_n_step = (n_step > 1)
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim=env.observation_space.shape[0],
                size=memory_size,
                batch_size=batch_size,
                n_step=n_step,
                gamma=gamma
            )
        else:
            self.n_step = 1
            self.memory_n = None
        
        # -----------------------------
        # Distributional DQN (C51)
        # -----------------------------
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        # Support is a tensor [v_min, ..., v_max] with `atom_size` bins
        self.support = torch.linspace(v_min, v_max, atom_size).to(self.device)
        
        # -----------------------------
        # Networks: Online & Target
        # -----------------------------
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.dqn = Network(
            in_dim=obs_dim,
            out_dim=action_dim,
            atom_size=self.atom_size,
            support=self.support
        ).to(self.device)

        self.dqn_target = Network(
            in_dim=obs_dim,
            out_dim=action_dim,
            atom_size=self.atom_size,
            support=self.support
        ).to(self.device)

        # Copy weights from the online net into the target net initially
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()  # We don't train the target net directly

        # -----------------------------
        # Optimizer
        # -----------------------------
        self.optimizer = optim.Adam(self.dqn.parameters())

        # We'll store transitions in self.transition for single steps
        self.transition = list()

        # Toggling train/test mode
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select an action for the current state using the online network.
        No epsilon-greedy is used here because NoisyNet handles exploration by design.
        """
        # Convert to tensor and run it through the DQN
        state_tensor = torch.FloatTensor(state).to(self.device)
        # The forward pass outputs Q-values => shape [batch_size, num_actions]
        # Argmax chooses the best action
        action = self.dqn(state_tensor).argmax().detach().cpu().numpy()

        if not self.is_test:
            # We'll store the (s, a, r, s', done) into memory eventually
            # This 'transition' list starts with (s, a)
            self.transition = [state, action]
        
        return action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Execute the selected action in the environment.
        Append the new transition info to the n-step buffer if needed.
        Then, if a full n-step transition is ready, store it in the prioritized buffer.
        """
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            # Append the rest of the transition tuple (reward, next_state, done)
            self.transition += [reward, next_state, done]

            # If n-step is enabled, we store in the n-step buffer first.
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            else:
                one_step_transition = self.transition

            # Once an n-step transition is completed, 'one_step_transition' is not None
            # We then store that transition in the main PER buffer.
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done

    def update_model(self) -> float:
        """
        Main training routine:
          1. Sample from prioritized buffer
          2. Compute distributional double-DQN loss
          3. Combine 1-step and n-step losses (if use_n_step=True)
          4. Apply importance sampling weights & backprop
          5. Update priorities
          6. Reset NoisyNet noise
        """
        # 1) Sample from prioritized replay using current beta
        samples = self.memory.sample_batch(beta=self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]  # for updating priorities later

        # 2) Compute 1-step distributional loss
        elementwise_loss = self._compute_dqn_loss(samples, gamma=self.gamma)

        # 3) If we also want n-step returns, combine them
        #    The original Rainbow typically used only n-step. 
        #    Here we do both for demonstration, so we sum their losses.
        loss = torch.mean(elementwise_loss * weights)
        if self.use_n_step:
            gamma_n = self.gamma ** self.n_step
            # We'll sample the same 'indices' from the n-step buffer
            samples_n = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n = self._compute_dqn_loss(samples_n, gamma=gamma_n)
            elementwise_loss += elementwise_loss_n
            # Weighted average
            loss = torch.mean(elementwise_loss * weights)

        # 4) Gradient update
        self.optimizer.zero_grad()
        loss.backward()
        # sometimes gradient clipping helps with stability
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # 5) Update priorities in the PER buffer
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        
        # 6) Reset NoisyNet noise in both online & target networks
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        # Return the scalar loss for logging
        return loss.item()

    def train(self, num_frames: int, plotting_interval: int = 200):
        """
        The main loop:
          - Repeatedly select actions and store transitions
          - Periodically sample a batch and update the network
          - Periodically update the target network
          - Track scores and losses
        """
        self.is_test = False
        
        # Start a new episode
        state, _ = self.env.reset(seed=self.seed)
        update_cnt = 0
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            # 1) Choose an action w.r.t current noisy net
            action = self.select_action(state)

            # 2) Step in the environment
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward
            
            # 3) Increase beta over time (PER)
            fraction = min(frame_idx / num_frames, 1.0)
            # self.beta moves from initial value towards 1.0
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # 4) If episode ends, reset environment
            if done:
                state, _ = self.env.reset(seed=self.seed)
                scores.append(score)
                score = 0

            # 5) Train if the replay buffer is large enough
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                
                # 6) Hard-update the target net at fixed intervals
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # 7) Optional: logging / plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses)
                
        # End of training
        self.env.close()
                
    def test(self, video_folder: str) -> None:
        """
        Evaluate the agent (no training). 
        Also records a video for demonstration.
        """
        self.is_test = True
        
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)
        
        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0
        
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward
        
        print("Score:", score)
        self.env.close()
        
        # restore environment
        self.env = naive_env

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """
        The core distributional DQN (C51) loss, combined with Double DQN logic for
        action selection and target distribution evaluation.

        Steps:
          1) next_action <- argmax_a Q(s', a; dqn)   (online net)
          2) next_dist   <- dist(s', next_action; dqn_target)  (target net)
          3) shift distribution by reward + gamma, then project onto [v_min, v_max]
          4) compute cross-entropy w.r.t the current dist(s, a; dqn)
        """
        device = self.device
        batch_size = samples["obs"].shape[0]

        # Tensors
        state     = torch.FloatTensor(samples["obs"]).to(device)
        next_state= torch.FloatTensor(samples["next_obs"]).to(device)
        action    = torch.LongTensor(samples["acts"]).to(device)
        reward    = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done      = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # We define some helpful constants for the distribution
        delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # 1) Double Q-Learning: 
            #    pick next_action from the 'online' network
            next_action = self.dqn(next_state).argmax(1)  # [batch_size]

            # 2) Evaluate that action with the target network's distribution
            next_dist = self.dqn_target.dist(next_state)  # [batch_size, act_dim, atom_size]
            next_dist = next_dist[range(batch_size), next_action] 
              # shape => [batch_size, atom_size]

            # 3) Construct the shifted/clamped support for the next state:
            t_z = reward + (1 - done) * gamma * self.support
            # keep them within [v_min, v_max]
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)

            # Convert these into integer “bin” indices
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            # offset for indexing
            offset = (
                torch.linspace(0, (batch_size - 1) * self.atom_size, batch_size)
                .long().unsqueeze(1).expand(batch_size, self.atom_size).to(device)
            )

            # Build projected distribution
            proj_dist = torch.zeros_like(next_dist)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1),
                (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1),
                (next_dist * (b - l.float())).view(-1)
            )

        # 4) Evaluate the online network’s distribution for (s, a)
        dist = self.dqn.dist(state)  # shape: [batch_size, act_dim, atom_size]
        log_p = torch.log(dist[range(batch_size), action])  # [batch_size, atom_size]

        # Cross-entropy or KL Divergence:
        #  sum over the atom dimension: CE(proj_dist, dist) = - sum( proj_dist * log(dist) )
        elementwise_loss = -(proj_dist * log_p).sum(dim=1)

        return elementwise_loss

    def _target_hard_update(self):
        """
        Copy the online network's parameters to the target network.
        """
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(self, frame_idx: int, scores: List[float], losses: List[float]):
        """
        Simple plotting for debugging/tracking.
        """
        from IPython.display import clear_output
        import matplotlib.pyplot as plt

        clear_output(True)
        plt.figure(figsize=(20, 5))

        plt.subplot(131)
        plt.title("frame %s. score: %s" % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)

        plt.subplot(132)
        plt.title("loss")
        plt.plot(losses)

        plt.show()
