import torch
from networks.actor import TradingActor
from networks.critic import TransformerCritic
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from networks.shared import TemporalTransformerEncoder
from typing import Optional, Dict
from datetime import datetime
import matplotlib
#matplotlib.use("Agg")  # Use a non-interactive backend to avoid Tkinter issues

class PPOBuffer:
    def __init__(self, buffer_size: int = 2048):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.buffer_size = buffer_size

    def store(self,
              state: np.ndarray,
              action: np.ndarray,
              reward: float,
              next_state: np.ndarray,
              done: bool,
              value: torch.Tensor,
              log_prob: torch.Tensor):
        """Store experience in buffer"""
        self.states.append(torch.FloatTensor(state))
        self.actions.append(torch.FloatTensor(action))
        self.rewards.append(reward)
        self.next_states.append(torch.FloatTensor(next_state))
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def clear(self):
        """Clear buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def __len__(self):
        return len(self.rewards)


class PPOAgent:
    def __init__(
        self,
        env,
        dataset: pd.DataFrame,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        batch_size: int = 128,
        ppo_epochs: int = 5,
        max_grad_norm: float = 1,
        lambda_cov: float = 1e0  # New hyperparameter for covariance penalty, if needed later.
    ):
        self.env = env
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        self.device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.ppo_epochs = ppo_epochs
        self.max_grad_norm = max_grad_norm
        self.lambda_cov = lambda_cov
        self.entropy_coef = 0.01  # Entropy coefficient for the loss bonus.
        self.feature_importance_history = []
        self.update_count = 0

        # NEW: Set context length (e.g. past month = 30 time steps) and state history for critic context.
        self.context_length = 63
        self.state_history = []  # will maintain recent states during an episode

        # Get initial observation to determine state_dim
        state = env.reset()
        state_dim = len(state)
        
        # Use the environment-provided feature names so they match the observation vector
        feature_names = env.get_feature_names()

        # Get action_dim from environment's action space
        action_dim = env.action_space.shape[0]

        # Create a shared temporal transformer encoder for both actor and critic.
        shared_temporal_encoder = TemporalTransformerEncoder(
                input_dim=state_dim,
                hidden_dim=128,
                num_heads=4,
                num_layers=2,
                max_seq_len=self.context_length,
                dropout=0.1
        )

        # Initialize networks with dynamic dimensions
        self.actor = TradingActor(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=actor_lr,
            temporal_encoder=shared_temporal_encoder,
            context_length=self.context_length
        ).to(self.device)
        
        # UPDATED: Use TransformerCritic (which expects sequence input) instead of PricingCritic.
        self.critic = TransformerCritic(
            state_dim=state_dim,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            context_length=self.context_length,
            lr=critic_lr,
            temporal_encoder=shared_temporal_encoder,
            action_dim=action_dim
        ).to(self.device)
        
        self.buffer = PPOBuffer()
        
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=f'./runs/feature_attention_{run_id}')
        self.dataset = dataset
        
    def get_action(self, state: np.ndarray) -> tuple:
        """
        Given the current state, return a sampled portfolio allocation (i.e. a probability vector),
        the log probability of that action, and the critic's value estimate.
        """
        # Convert input state to tensor and move it to the device.
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)

        # Update the state history to create a temporal context sequence.
        if not self.state_history:
            self.state_history = [state_tensor for _ in range(self.context_length)]
        else:
            self.state_history.append(state_tensor)
            if len(self.state_history) > self.context_length:
                self.state_history = self.state_history[-self.context_length:]

        # Build the state sequence tensor: (batch=1, context_length, state_dim)
        state_seq = torch.stack(self.state_history).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Use actor.sample to obtain action, log probability, etc.
            allocation, log_prob, _ = self.actor.sample(state_tensor, state_seq=state_seq)

            # Pass both state sequence and actions (now batched) to the critic.
            value = self.critic(state_seq, actor=self.actor).squeeze()

        return allocation.cpu().numpy(), log_prob.cpu(), value.cpu()


    # NEW HELPER: Build a sliding-window sequence tensor from a series of state tensors.
    def _build_state_sequences(self, states: torch.Tensor, context_length: int) -> torch.Tensor:
        """
        Given states of shape (T, state_dim), returns a tensor of shape 
        (T, context_length, state_dim) where each row t contains the 
        sequence [states[t - context_length + 1], ..., states[t]] with padding
        at the beginning (using the first state).
        """
        T, state_dim = states.shape
        seqs = []
        for i in range(T):
            if i < context_length:
                # Use pad with exactly (context_length - (i+1)) copies of the first state
                pad = states[0].unsqueeze(0).repeat(context_length - (i + 1), 1)
                seq = torch.cat([pad, states[: i + 1]], dim=0)
            else:
                seq = states[i - context_length + 1 : i + 1]
            seqs.append(seq)
        return torch.stack(seqs)  # (T, context_length, state_dim)

    def update(self):
        # Initialize loss trackers
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        
        # Convert buffered values to tensors.
        old_values = torch.tensor(np.stack(self.buffer.values).squeeze(), dtype=torch.float32).to(self.device)
        old_log_probs = torch.stack(self.buffer.log_probs).to(self.device)
        
        # Convert state buffers and build state sequences.
        states = torch.stack(self.buffer.states).to(self.device)  # shape: (T, state_dim)
        state_seqs = self._build_state_sequences(states, self.context_length)
        actions = torch.stack(self.buffer.actions).to(self.device)
        rewards = torch.tensor(np.array(self.buffer.rewards), dtype=torch.float32).to(self.device)
        next_states = torch.stack(self.buffer.next_states).to(self.device)
        next_state_seqs = self._build_state_sequences(next_states, self.context_length)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32).to(self.device)

        # Calculate advantages and returns.
        returns = []
        advantages = []
        last_value = self.critic(next_state_seqs[-1].unsqueeze(0), actor=self.actor).squeeze().detach()
        advantage = 0
        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else old_values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - old_values[t]
            advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)
            returns.insert(0, advantage + old_values[t])
        
        advantages = torch.stack(advantages).to(self.device)
        returns = torch.stack(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO minibatch update loop
        for _ in range(self.ppo_epochs):
            for indices in self._get_batch_indices():
                batch_state_seqs = state_seqs[indices]  # (batch, context_length, state_dim)
                batch_states = states[indices]          # (batch, state_dim)
                batch_actions = actions[indices]          # (batch, action_dim)
                batch_returns = returns[indices]
                batch_advantages = advantages[indices]
                batch_old_log_probs = old_log_probs[indices]
                
                with torch.enable_grad():
                    # The actor uses the batch states to construct a distribution.
                    dist = self.actor.get_distribution(batch_states, state_seq=batch_state_seqs)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy_mean = dist.entropy().mean()
                    # Use the critic by passing both the batch state sequences and batch actions.
                    values = self.critic(batch_state_seqs, actor=self.actor).squeeze(-1)
                
                # Calculate the probability ratio.
                ratio = (new_log_probs - batch_old_log_probs).exp()
                
                policy_loss1 = ratio * batch_advantages
                policy_loss2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
                value_loss = 0.5 * (batch_returns - values).pow(2).mean()
                
                # Total loss includes an entropy bonus (subtracted to encourage exploration).
                loss = policy_loss + value_loss - self.entropy_coef * entropy_mean
                
                self.actor.optimizer.zero_grad(set_to_none=True)
                self.critic.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                
                # Log gradient norms BEFORE clipping for debugging purposes.
                for name, param in self.actor.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm()
                        self.writer.add_scalar(f"Gradient Norms/Actor/{name}", grad_norm.item(), self.update_count)
                for name, param in self.critic.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm()
                        self.writer.add_scalar(f"Gradient Norms/Critic/{name}", grad_norm.item(), self.update_count)
                
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
                actor_losses.append(policy_loss.item())
                critic_losses.append(value_loss.item())
                entropy_losses.append(entropy_mean.item())
        
        mean_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
        mean_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        mean_entropy_loss = np.mean(entropy_losses) if entropy_losses else 0.0

        # Add feature analysis after update
        with torch.no_grad():
            if len(self.buffer.states) > 0:
                sample_state = self.buffer.states[0].to(self.device)
                _ = self.actor(sample_state.unsqueeze(0))
                analysis = self.get_feature_analysis()
                if analysis is not None:
                    # Updated logging using aggregated feature keys.
                    for name, weight in zip(analysis["aggregated_feature_names"], analysis["aggregated_attention_weights"]):
                        self.writer.add_scalar(f'Feature Importance/{name}', weight, self.update_count)
                    
                    # Log the elbow plot to visualize the ordered importances and cutoff.
                    self.writer.add_figure(
                        'Feature Importance Elbow Plot',
                        self._create_importance_elbow_plot(analysis),
                        self.update_count
                    )
                    
                    # Log which features have very low aggregated importance.
                    self.writer.add_text(
                        'Features/Elimination Candidates', 
                        ', '.join(analysis["elimination_candidates"]),
                        self.update_count
                    )
                    
                    # Save the aggregated importance history for potential later analysis.
                    self.feature_importance_history.append(analysis["aggregated_attention_weights"])
                    self.update_count += 1

        # Flush the writer to ensure logs are written to disk
        self.writer.flush()

        # Clear buffer after update
        self.buffer.clear()

        return mean_actor_loss, mean_critic_loss, mean_entropy_loss

    def _get_batch_indices(self):
        """Handle cases where buffer size < mini_batch_size"""
        buffer_size = len(self.buffer)
        mini_batch_size = min(buffer_size, self.batch_size)
        indices = torch.randperm(buffer_size)
        for i in range(0, buffer_size, mini_batch_size):
            yield indices[i:i + mini_batch_size] 

    def get_feature_analysis(self) -> Optional[Dict]:
        """Get aggregated feature importance analysis using gradient saliency of actor's output.
        
        This method computes the saliency for each input feature by averaging the buffered states,
        forwarding through the actor (with a dummy temporal sequence if used) and then computing the
        absolute gradient with respect to the input. For features that pertain to the same metric
        across different stocks (e.g. 'AAPL_volume', 'MSFT_volume'), the saliencies are aggregated
        by computing the L2 norm across the stock-specific features. An elbow method is applied on the
        aggregated importances to determine candidate metrics for elimination.
        
        Returns:
            A dictionary with keys:
              - "aggregated_feature_names": list of aggregated metric names (e.g. "volume").
              - "aggregated_attention_weights": aggregated (L2 norm) importance per metric.
              - "aggregation_mapping": mapping from each aggregated metric to the list of original feature names.
              - "elimination_candidates": aggregated metrics with importance below an adaptive threshold.
        """
        if not self.buffer.states:
            return None

        with torch.enable_grad():
            # Compute representative mean state with gradient tracking.
            sample_state = torch.stack(self.buffer.states).to(self.device)  # (T, state_dim)
            sample_state = sample_state.mean(dim=0, keepdim=True)             # (1, state_dim)
            sample_state = sample_state.clone().detach().requires_grad_(True)

            # Create dummy temporal sequence if temporal context is used.
            if self.context_length is not None:
                dummy_seq = sample_state.repeat(self.context_length, 1).unsqueeze(0)  # (1, context_length, state_dim)
            else:
                dummy_seq = None

            # Forward pass through the actor.
            output = self.actor.forward(sample_state, dummy_seq)
            loss = output.sum()
            loss.backward()

            # Compute normalized absolute gradient saliency per feature.
            saliency = sample_state.grad.abs().squeeze(0)  # (state_dim,)
            normalized_saliency = saliency / (saliency.sum() + 1e-8)
        salience_np = normalized_saliency.detach().cpu().numpy()

        # Retrieve the feature names from the environment.
        feature_names = self.env.get_feature_names()

        # Aggregate features by metric using L2 norm instead of mean.
        # Assume feature names are formatted as "stock_metric" (e.g. "AAPL_volume").
        aggregated_importance: Dict[str, list] = {}
        aggregation_mapping: Dict[str, list] = {}
        for idx, fname in enumerate(feature_names):
            parts = fname.split("_", 1)
            metric = parts[1] if len(parts) == 2 else fname
            aggregated_importance.setdefault(metric, []).append(salience_np[idx])
            aggregation_mapping.setdefault(metric, []).append(fname)

        # Compute aggregated (L2-norm) saliency per metric.
        agg_metrics = []
        agg_values = []
        for metric, values in aggregated_importance.items():
            norm_value = np.linalg.norm(np.array(values))
            agg_metrics.append(metric)
            agg_values.append(norm_value)

        # Sort metrics by descending aggregated importance.
        sorted_tuples = sorted(zip(agg_metrics, agg_values), key=lambda x: x[1], reverse=True)
        aggregated_feature_names = [x[0] for x in sorted_tuples]
        aggregated_attention_weights = np.array([x[1] for x in sorted_tuples])

        # Apply the elbow method on the aggregated importances.
        if len(aggregated_attention_weights) >= 2:
            differences = np.diff(aggregated_attention_weights)
            elbow_index = int(np.argmax(differences))
            adaptive_threshold = (
                aggregated_attention_weights[elbow_index + 1]
                if (elbow_index + 1) < len(aggregated_attention_weights)
                else 0
            )
        else:
            adaptive_threshold = 0

        elimination_candidates = [
            metric
            for metric, att in zip(aggregated_feature_names, aggregated_attention_weights)
            if att < adaptive_threshold
        ]

        return {
            "aggregated_feature_names": aggregated_feature_names,
            "aggregated_attention_weights": aggregated_attention_weights,
            "aggregation_mapping": aggregation_mapping,
            "elimination_candidates": elimination_candidates,
        }

    def _create_attention_heatmap(self, analysis):
        """Creates a heatmap visualization of feature attention weights"""
        plt.figure(figsize=(12, 4))
        sns.heatmap(
            analysis["attention_weights"].reshape(1, -1),
            xticklabels=analysis["feature_names"],
            yticklabels=False,
            cmap='YlOrRd',
            annot=True,
            fmt='.3f'
        )
        plt.title('Feature Attention Weights')
        plt.xticks(rotation=45, ha='right')
        return plt.gcf()

    def _create_importance_elbow_plot(self, analysis):
        """
        Creates a plot of sorted aggregated feature importances with an indication
        of the adaptive cutoff (elbow) for candidate elimination.

        Args:
            analysis (dict): Dictionary containing aggregated metrics with keys:
                    - "aggregated_feature_names": list of aggregated metric names.
                    - "aggregated_attention_weights": aggregated importance values.
        Returns:
            Matplotlib figure.
        """
        import matplotlib.pyplot as plt

        agg_names = analysis["aggregated_feature_names"]
        agg_importance = analysis["aggregated_attention_weights"]

        if len(agg_importance) < 2:
            return None  # Not enough metrics to plot.

        differences = np.diff(agg_importance)
        elbow_index = int(np.argmax(differences))
        adaptive_threshold = (
            agg_importance[elbow_index + 1] if (elbow_index + 1) < len(agg_importance) else 0
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            range(1, len(agg_importance) + 1),
            agg_importance,
            marker="o",
            label="Sorted Aggregated Importance",
        )
        ax.axhline(
            y=adaptive_threshold,
            color="r",
            linestyle="--",
            label=f"Adaptive Threshold: {adaptive_threshold:.2e}",
        )
        ax.axvline(
            x=elbow_index + 1,
            color="g",
            linestyle="--",
            label=f"Elbow at rank: {elbow_index + 1}",
        )
        ax.set_xlabel("Metric Rank")
        ax.set_ylabel("Aggregated Normalized Importance")
        ax.set_title("Elbow Detection for Aggregated Feature Importance")
        ax.legend()
        plt.tight_layout()
        return fig

    def close(self):
        """Call when finished training to close TensorBoard writer"""
        self.writer.close()

    