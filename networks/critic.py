import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Optional
from networks.shared import TemporalTransformerEncoder

def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=nn.init.calculate_gain('relu'))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.TransformerDecoderLayer):
        nn.init.xavier_uniform_(module.linear1.weight)
        nn.init.xavier_uniform_(module.linear2.weight)
        if module.linear1.bias is not None:
            nn.init.zeros_(module.linear1.bias)
        if module.linear2.bias is not None:
            nn.init.zeros_(module.linear2.bias)

class TransformerCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        context_length: int = 63,
        forecast_horizon: int = 10,  # Number of simulation steps into the future.
        lr: float = 3e-4,
        gamma: float = 0.99,  # Canonical discount factor (set from the environment).
        temporal_encoder: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        action_dim: Optional[int] = None  # Dimension of actor actions, if conditioning desired.
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.gamma = gamma  # Canonical discount factor
        if temporal_encoder is None:
            self.temporal_encoder = TemporalTransformerEncoder(
                input_dim=state_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                max_seq_len=context_length,
            )
        else:
            self.temporal_encoder = temporal_encoder

        # When an action dimension is provided, we condition on the simulated actor action.
        if action_dim is not None:
            self.actor_embed = nn.Linear(action_dim, hidden_dim)
        else:
            self.actor_embed = None

        # Learnable decoder queries—each used for one simulation step.
        self.decoder_query = nn.Parameter(torch.randn(forecast_horizon, hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout_rate, activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Final projection to obtain a scalar value from the decoder's output.
        self.out_proj = nn.Linear(hidden_dim, 1)
        # Projection to update the simulated state back to the raw state space.
        self.state_projection = nn.Linear(hidden_dim, state_dim)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.apply(initialize_weights)

    def forward(
        self, 
        state_seq: torch.Tensor,  # (batch, context_length, state_dim)
        actor: nn.Module
    ) -> torch.Tensor:
        """
        Simulates the expected future value by iteratively predicting discounted future state values.
        The simulation procedure is as follows:
        
          1. Predict the value of the current state.
          2. Use the actor (current policy) to simulate an action given the current state.
          3. Condition the decoder query with the simulated action's embedding.
          4. Predict the discounted value of the next state.
          5. Update the state estimate using a learned state projection.
          6. Iterate for the forecast horizon and sum the discounted predicted values.
        
        Parameters:
            state_seq (torch.Tensor): Sequence of raw states (batch, context_length, state_dim)
            actor (nn.Module): The current policy actor used to simulate actions. Expects both
                               current_state and state_seq as inputs.
        
        Returns:
            torch.Tensor: Expected future value for each state in the batch (batch,).
        """
        # Encode the input sequence for context.
        encoded = self.temporal_encoder(state_seq)  # (batch, context_length, hidden_dim)
        memory = encoded.transpose(0, 1)  # (context_length, batch, hidden_dim)
        batch_size = state_seq.size(0)
        # Start simulation at the current state (last state in the sequence).
        current_state = state_seq[:, -1, :]  # (batch, state_dim)
        total_value = torch.zeros(batch_size, device=state_seq.device)
        discount = 1.0

        # Iteratively simulate over the forecast horizon.
        for i in range(self.forecast_horizon):
            # Use no_grad to ensure gradients don't flow into the actor.
            with torch.no_grad():
                allocation = actor.get_allocation(current_state, state_seq)  # (batch, action_dim)
            action_feat = (
                self.actor_embed(allocation)
                if self.actor_embed is not None
                else torch.zeros(batch_size, self.decoder_query.size(-1), device=state_seq.device)
            )

            # Prepare the decoder query for this simulation step.
            query = self.decoder_query[i].unsqueeze(0).expand(batch_size, -1)  # (batch, hidden_dim)
            # Condition the query with the embedded actor allocation.
            query = query + action_feat
            # Transformer decoder requires (tgt_seq_len, batch, hidden_dim); here tgt_seq_len is 1.
            query = query.unsqueeze(0)  # (1, batch, hidden_dim)
            decoded = self.transformer_decoder(tgt=query, memory=memory)  # (1, batch, hidden_dim)
            decoded_token = decoded[0]  # (batch, hidden_dim)

            # Predict the scalar value for this simulated step.
            value_step = self.out_proj(decoded_token).squeeze(-1)  # (batch,)
            total_value += discount * value_step
            discount *= self.gamma

            # Update the simulated current state (project the hidden state back to state space).
            current_state = self.state_projection(decoded_token)  # (batch, state_dim)
            # Update the state sequence by rolling it forward:
            # Drop the oldest state and append the newly predicted state.
            state_seq = torch.cat([state_seq[:, 1:, :], current_state.unsqueeze(1)], dim=1)

        return total_value  