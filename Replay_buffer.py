import math
import os
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_

from segment_tree import MinSegmentTree, SumSegmentTree

from collections import deque
import numpy as np
import random
from typing import Dict, Tuple, Deque, List

class ReplayBuffer:
    """
    This buffer holds transitions for training a DQN.
    It supports optional n-step returns to speed up learning.
    """

    def __init__(
        self, 
        obs_dim: int,         # Dimensionality of the state space
        size: int,            # Maximum number of transitions to store
        batch_size: int = 32, 
        n_step: int = 1,      # n-step parameter
        gamma: float = 0.99   # discount factor
    ):
        # Preallocate numpy arrays for states, next states, actions, rewards, done flags
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.max_size = size       # Max capacity of the buffer
        self.batch_size = batch_size

        # Pointer to where the next transition will be stored
        self.ptr = 0
        # Current number of valid entries in the buffer
        self.size = 0

        # For n-step returns
        # We'll collect transitions in a small deque (maxlen=n_step)
        # and only once we have n transitions do we combine them into one
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self, 
        obs: np.ndarray, 
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """
        Store one transition. If we have enough transitions for n-step, 
        we compute and store the resulting multi-step transition in the main buffers.
        Returns the earliest single-step transition in the n-step buffer, 
        or an empty tuple if n-step isn't ready yet.
        """
        # Put current single-step transition in the n-step buffer
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # If we haven't yet collected n transitions, we can't form an n-step transition
        if len(self.n_step_buffer) < self.n_step:
            return ()  # Return empty, no new entry in main buffer

        # Now we have n transitions in the deque => create one n-step transition
        # that sums rewards over n steps and leaps forward to the state after n steps.
        rew, next_obs, done = self._get_n_step_info(self.n_step_buffer, self.gamma)
        # The initial (s0, a0) come from the *first* transition in the deque
        obs, act = self.n_step_buffer[0][:2]

        # Store that n-step transition into the main buffer arrays
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        # Move the insertion pointer
        self.ptr = (self.ptr + 1) % self.max_size
        # Increase the size, but cap at max_size
        self.size = min(self.size + 1, self.max_size)
        
        # Return the earliest transition for debugging or further usage
        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """
        Sample a batch uniformly at random from the buffer.
        Returns a dict with keys: obs, next_obs, acts, rews, done, indices.
        """
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return {
            "obs":      self.obs_buf[idxs],
            "next_obs": self.next_obs_buf[idxs],
            "acts":     self.acts_buf[idxs],
            "rews":     self.rews_buf[idxs],
            "done":     self.done_buf[idxs],
            # We also return these indices, which can be useful 
            # if we want to update priorities later or do debugging.
            "indices":  idxs,
        }
    
    def sample_batch_from_idxs(self, idxs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Same as sample_batch, but we allow specifying the exact indices to gather 
        (used when combining 1-step and n-step losses, for example).
        """
        return {
            "obs":      self.obs_buf[idxs],
            "next_obs": self.next_obs_buf[idxs],
            "acts":     self.acts_buf[idxs],
            "rews":     self.rews_buf[idxs],
            "done":     self.done_buf[idxs],
        }
    
    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.float32, np.ndarray, bool]:
        """
        Given n single-step transitions in n_step_buffer, 
        compute the multi-step return, next_obs, and done flag.
        
        We do this by starting from the last transition and going backwards, 
        repeatedly discounting the reward and seeing if the done flag is set.
        """
        # Last transition in the buffer => immediate rew, next_obs, done
        rew, next_obs, done = n_step_buffer[-1][-3:]

        # Move backwards through the first (n-1) transitions
        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]
            # If the environment was done at step k, 
            # we don't want subsequent rewards or states
            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        """
        Return the current size (number of valid entries) in the main buffer.
        """
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Inherits from ReplayBuffer but adds priority-based sampling using segment trees.
    This means each transition has a priority p_i, 
    and we sample transitions proportionally to p_i^alpha.
    """

    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6,
        n_step: int = 1, 
        gamma: float = 0.99,
    ):
        """
        alpha: how strongly we favor high-priority transitions 
               (alpha=0 => uniform sampling)
        """
        super(PrioritizedReplayBuffer, self).__init__(
            obs_dim, size, batch_size, n_step, gamma
        )
        self.max_priority = 1.0  # We start with 1.0 as the initial max priority
        self.tree_ptr = 0
        self.alpha = alpha
        
        # Build segment trees for sum and min. 
        # They help in O(log n) sampling and priority updates.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
        self, 
        obs: np.ndarray, 
        act: int, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """
        Same logic as the parent store method, but after storing the new transition, 
        we also set priority in the trees to max_priority for this index.
        """
        transition = super().store(obs, act, rew, next_obs, done)
        
        # If a transition was actually stored (meaning we had at least n steps):
        if transition:
            # Assign the maximum priority to the newly inserted transition
            # so that it has a high chance to be sampled soon
            idx = self.tree_ptr
            priority = self.max_priority ** self.alpha
            self.sum_tree[idx] = priority
            self.min_tree[idx] = priority

            # Advance tree_ptr in a circular manner
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        
        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """
        Sample transitions using the segment tree for priorities.
        beta: importance sampling exponent; if 0 => no IS correction, if 1 => full correction.
        """
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        # Gather transitions from those indices
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]

        # Compute importance sampling weights for each sampled index
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,  # important for scaling loss
            indices=indices,  # we need indices to update priorities later
        )
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """
        After computing new TD errors (or distributional divergences), 
        we set the new priorities for each transition.
        """
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            # Weighted by alpha
            p = priority ** self.alpha
            self.sum_tree[idx] = p
            self.min_tree[idx] = p

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """
        The 'proportional' variant of prioritized replay:
        we divide the total priority sum into N segments, 
        and pick one random value in each segment to retrieve an index.
        """
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            # retrieve(...) finds which index corresponds to the prefix-sum = upperbound
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """
        Compute the importance-sampling weight for a given sampled index, 
        scaled by the smallest sampling probability (so that w <= 1).
        """
        # smallest priority fraction
        p_min = self.min_tree.min() / self.sum_tree.sum()
        # maximum possible weight
        max_weight = (p_min * len(self)) ** (-beta)
        
        # probability of sampling index idx
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        # unscaled weight
        weight = (p_sample * len(self)) ** (-beta)
        # normalize by max_weight to ensure w <= 1
        weight /= max_weight
        
        return weight
