import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Dirichlet
import numpy as np
from typing import Optional


def atanh(x: torch.Tensor) -> torch.Tensor:
    """Numerically safe inverse hyperbolic tangent."""
    eps = 1e-6
    x = x.clamp(-1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class TradingActor(nn.Module):
    """
    TradingActor now outputs a portfolio allocation distribution rather than a mean action.
    The network produces logits that are transformed into allocation probabilities via softmax.
    These probabilities are then scaled to form the concentration parameters for a Dirichlet distribution.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        temporal_encoder: Optional[nn.Module] = None,
        context_length: Optional[int] = None
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.context_length = context_length
        self.temporal_encoder = temporal_encoder

        # Base network to process the current state.
        self.base_net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        # If a temporal encoder is provided, combine current state features with temporal context.
        if self.temporal_encoder is not None and self.context_length is not None:
            self.combination_layer = nn.Sequential(
                nn.Linear(128 + 128, 128),  # Concatenating base (128) and temporal features (128)
                nn.GELU()
            )

        # Allocation head: outputs logits for portfolio allocation. The softmax over these logits
        # yields the proportions of the portfolio's balance to invest in each stock.
        self.allocation_head = nn.Linear(128, action_dim)
        nn.init.xavier_uniform_(self.allocation_head.weight)
        nn.init.zeros_(self.allocation_head.bias)

        self.optimizer = Adam(self.parameters(), lr=lr)

        # Concentration scale modulates the confidence of the portfolio allocation.
        # A higher value yields lower variance (more confident allocation decisions).
        self.log_concentration_scale = nn.Parameter(torch.log(torch.tensor(10.0, dtype=torch.float)))

    def forward(self, state: torch.Tensor, state_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass that computes allocation logits from the input state (and optional temporal context).

        Parameters:
            state (torch.Tensor): Current state with shape (batch, state_dim)
            state_seq (Optional[torch.Tensor]): Temporal context with shape (batch, seq_length, feature_dim)

        Returns:
            torch.Tensor: Raw logits for portfolio allocation with shape (batch, action_dim).
                        These are later converted into allocation probabilities via softmax.
        """
        base_features = self.base_net(state)  # (batch, 128)
        if self.temporal_encoder is not None and state_seq is not None:
            temporal_features = self.temporal_encoder(state_seq)  # (batch, context_length, 128)
            # Use the last temporal token as the context representation.
            temp_feature = temporal_features[:, -1, :]  # (batch, 128)
            combined = torch.cat([base_features, temp_feature], dim=1)  # (batch, 256)
            features = self.combination_layer(combined)  # (batch, 128)
        else:
            features = base_features

        # Output logits (do not apply softmax here to allow computing gradients stably).
        logits = self.allocation_head(features)  # (batch, action_dim)
        return logits

    def get_allocation(self, state: torch.Tensor, state_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes portfolio allocation probabilities via a softmax over allocation logits.

        Parameters:
            state (torch.Tensor): Current state with shape (batch, state_dim)
            state_seq (Optional[torch.Tensor]): Temporal context

        Returns:
            torch.Tensor: Allocation probabilities (batch, action_dim) that sum to 1.
        """
        logits = self.forward(state, state_seq)
        allocation = F.softmax(logits, dim=-1)
        return allocation

    def get_distribution(self, state: torch.Tensor, state_seq: Optional[torch.Tensor] = None) -> Dirichlet:
        """
        Constructs a Dirichlet distribution from portfolio allocation probabilities.

        The softmax over allocation logits gives a baseline probability vector.
        Scaling by self.concentration_scale produces the concentration parameters for the Dirichlet.

        Parameters:
            state (torch.Tensor): Current state
            state_seq (Optional[torch.Tensor]): Temporal context

        Returns:
            Dirichlet: A Dirichlet distribution representing portfolio allocations.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        allocation = self.get_allocation(state, state_seq)  # (batch, action_dim)
        scale = torch.exp(self.log_concentration_scale)  # Ensures the scale stays positive.
        concentration = allocation * scale + 1e-3  # Ensure strict positivity
        return Dirichlet(concentration)

    def sample(self, state: torch.Tensor, state_seq: Optional[torch.Tensor] = None) -> tuple:
        """
        Samples a portfolio allocation from the Dirichlet distribution.

        Parameters:
            state (torch.Tensor): Current state with shape (batch, state_dim)
            state_seq (Optional[torch.Tensor]): Temporal context

        Returns:
            tuple: (allocation, log_prob, entropy) where:
              - allocation: A probability vector (sums to 1) representing the portfolio allocation.
              - log_prob: The log probability of the sampled allocation.
              - entropy: The entropy of the Dirichlet distribution (averaged if batched).
        """
        dist = self.get_distribution(state, state_seq)
        allocation = dist.rsample()
        log_prob = dist.log_prob(allocation)
        entropy = dist.entropy().mean()
        if allocation.shape[0] == 1:
            allocation = allocation[0]
            log_prob = log_prob[0]
        return allocation, log_prob, entropy

    def log_prob(self, state: torch.Tensor, action: torch.Tensor, state_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the log probability of a given portfolio allocation.

        Parameters:
            state (torch.Tensor): Current state.
            action (torch.Tensor): Portfolio allocation (probability vector).
            state_seq (Optional[torch.Tensor]): Temporal context.

        Returns:
            torch.Tensor: Log probability of the provided action.
        """
        dist = self.get_distribution(state, state_seq)
        return dist.log_prob(action)

