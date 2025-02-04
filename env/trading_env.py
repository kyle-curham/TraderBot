import gym
import numpy as np
from gym import spaces
from typing import Tuple, List
import torch


class StockTradingEnv(gym.Env):
    def __init__(self, processed_dataset, **kwargs):
        """Now expects pre-processed data from TradingDataProcessor"""

        # Simplified initialization
        self.dataset = processed_dataset
        self.n_stocks = len(processed_dataset.tic.unique())
        self.time_idx = processed_dataset.time_idx

        # Define feature groups based on processed data statistics:
        # Stock-level features (each tic)
        self.stock_cols = [
            'open', 'high', 'low', 'close', 'volume', 'vwap',
            'log_return', 'hl_spread', 'co_change', 'market_return', 'alpha',
            'vol_21d_ann', 'realized_vol_5d', 'parkinson_vol', 'vol_ratio_5d_21d',
            'volume_z', 'relative_volume', 'volume_momentum_5d', 'volume_volatility_21d',
            'trade_count'
        ]
        # System-level features (global features at each time index)
        self.system_cols = [
            "Real GDP", "CPI Inflation", "Unemployment Rate", "Fed Funds Rate",
            "S&P 500", "VIX Volatility", "10-Year Treasury Yield",
            "Industrial Production", "Retail Sales",
            "S&P 500_delta", "VIX Volatility_delta", "10-Year Treasury Yield_delta",
            "macd", "rsi_30", "close_30_sma", "close_60_sma",
            "turbulence"
        ]
        # When precomputing a features tensor, we flatten only the selected columns.
        self.features = processed_dataset[self.stock_cols + self.system_cols].values

        # System features dimension: number of system-level features.
        self.n_system_features = len(self.system_cols)
        
        # Initialize positions array with correct size.
        self.positions = np.zeros(self.n_stocks)

        # 1. INITIALIZE CRITICAL ATTRIBUTES FIRST
        self.initial_balance = kwargs.get('initial_balance', 1e4)
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.max_steps = self.dataset.groupby('tic').size().max() - 1
        self.current_step = 0 

        # Convert to numpy arrays for faster access
        self.market_returns = processed_dataset.groupby('time_idx')['market_return'].first().values
        self.market_volatility = processed_dataset.groupby('time_idx')['vol_21d_ann'].first().values

        # THEN initialize turbulence calculations
        self.turbulence_ary = processed_dataset.groupby('time_idx')['turbulence'].first().values
        self.turbulence_bool = (self.turbulence_ary > kwargs.get('turbulence_thresh', 0.9)).astype(float)
        
        # Add critical financial parameters
        self.turbulence_thresh = kwargs.get('turbulence_thresh', 0.9)
        # Canonical discount factor used by the environment, agent, and critic.
        self.gamma = kwargs.get('gamma', 0.99)
        self.buy_cost_pct = kwargs.get('buy_cost_pct', 0)
        self.sell_cost_pct = kwargs.get('sell_cost_pct', 0)
        self.reward_scaling = kwargs.get('reward_scaling', 1.0)
        self.max_stock = kwargs.get('max_stock', 1e1)
        self.min_stock_rate = kwargs.get('min_stock_rate', 0.9)
        
        # Introduce reward_horizon (fixed-length reward history for decoder targets)
        self.reward_horizon = kwargs.get('reward_horizon', 10)

        # Precompute features tensor if needed.
        self.feature_tensor = torch.tensor(self.features, dtype=torch.float32)
        self.device = kwargs.get('device', torch.device('cpu'))

        # 2. SET UP ACTION/OBSERVATION SPACES
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0,
            shape=(self.n_stocks,),
            dtype=np.float32
        )

        # Observation Space dimensions:
        # Financial state: 4 features
        financial_dim = 4  
        # Stock features: defined per ticker.
        stock_features_per_ticker = len(self.stock_cols)  
        # System features: from self.system_cols.
        system_features_dim = len(self.system_cols)  

        total_dim = financial_dim + (stock_features_per_ticker * self.n_stocks) + system_features_dim
        
        self.observation_space = spaces.Box(
            low=-3000, high=3000,
            shape=(total_dim,),
            dtype=np.float32
        )

        # Initialize return history
        self.return_history = []

        # Store price normalization parameters (for denormalization in _current_prices)
        self.close_mins = processed_dataset.groupby('tic')['close_min'].first().values
        self.close_maxs = processed_dataset.groupby('tic')['close_max'].first().values

    def reset(self) -> np.ndarray:
        """Reset environment state and return actor's observation vector"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = np.zeros(self.n_stocks)
        self.portfolio_value = self.initial_balance
        self.rewards_history = []  # Holds rewards for up to self.reward_horizon steps.
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Full observation vector with all features"""
        # Financial state (4 features)
        financial_state = np.array([
            self.balance/self.initial_balance,
            self.current_step / self.max_steps,
            self.market_returns[self.current_step],
            self.market_volatility[self.current_step]
        ])
        
        # Stock features: (stock_cols per tic)
        grouped = self.dataset.groupby('tic')
        stock_data = grouped.nth(self.current_step)[self.stock_cols]
        stock_features = stock_data.values.flatten()

        # System features: (global features for the current time index)
        # We assume one row per time index. Extract the row (using time_idx filter)
        time_slice = self.dataset.time_idx == self.current_step
        system_features = self.dataset.loc[time_slice, self.system_cols].values[0]

        return np.concatenate([
            financial_state,
            stock_features,
            system_features
        ]).astype(np.float32)

    def _financial_state(self) -> np.ndarray:
        return np.array([
            self.balance/self.initial_balance,
            self.turbulence_ary[self.current_step],
            self.current_step / self.max_steps,
            self.market_returns[self.current_step],
            self.market_volatility[self.current_step],
        ])

    def _compute_discounted_reward(self, bootstrap_value: float = 0.0) -> np.ndarray:
        """
        Computes the sequence of discounted returns over the fixed reward horizon based on the
        stored rewards and a provided bootstrap value for missing future rewards.
        
        For each future offset i, the target is computed as:
        
            If i < len(rewards_history):
                target[i] = (r_i + gamma*r_{i+1} + ... + gamma^(n-i-1)*r_{n-1})
                            + gamma^(n-i) * bootstrap_value
            Else:
                target[i] = gamma^(i-n) * bootstrap_value  (with i-n >= 0)
                
        This produces a vector of length self.reward_horizon.
        
        Returns:
            np.ndarray: A vector of discounted return targets of shape (self.reward_horizon,).
        """
        H = self.reward_horizon
        n = len(self.rewards_history)
        discounted_target = np.zeros(H)
        for i in range(H):
            if i < n:
                cumulative = 0.0
                for j in range(n - i):
                    cumulative += (self.gamma ** j) * self.rewards_history[i + j]
                cumulative += (self.gamma ** (n - i)) * bootstrap_value
                discounted_target[i] = cumulative
            else:
                # For time steps in the horizon where no reward has been observed yet,
                # use the bootstrapped value with appropriate discount.
                discounted_target[i] = (self.gamma ** (i - n)) * bootstrap_value
        return discounted_target.sum()

    def step(self, actions: np.ndarray, bootstrap_value: float = 0.0) -> Tuple[np.ndarray, float, bool, dict]:
        # Execute trades first
        self._execute_trades(actions)
        
        # Update portfolio value before getting new state
        current_value = self.balance + np.sum(self.positions * self._current_prices())
        if current_value <= 0:
            breakpoint()
        # Compute immediate reward as the log return
        immediate_return = np.log(current_value / self.portfolio_value) if self.portfolio_value > 0 else 0.0
        # Scale the immediate reward
        scaled_reward = immediate_return * self.reward_scaling
        # Append immediate reward and maintain fixed-length reward history
        self.rewards_history.append(scaled_reward)
        if len(self.rewards_history) > self.reward_horizon:
            self.rewards_history.pop(0)
        
        # Compute discounted rewards over the horizon at every timestep
        discounted_reward = self._compute_discounted_reward(bootstrap_value=bootstrap_value)
        
        # Update portfolio value reference for next step
        self.portfolio_value = current_value
        
        # Advance environment step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        info = {
            'normalized_portfolio': current_value * self.reward_scaling,
            'normalized_balance': self.balance,
            'raw_value': current_value,
            'discounted_rewards': discounted_reward
        }
        
        return self._get_obs(), discounted_reward, done, info

    def _execute_trades(self, actions: np.ndarray) -> None:
        """Rebalance portfolio allocations based on desired proportions.

        This method computes the target number of shares for each stock based on the desired
        portfolio allocation and current portfolio value (cash + holdings). It then executes a two-phase
        reallocation:
          1. SELL PHASE: If current holdings exceed the target, sell the excess shares.
          2. BUY PHASE: If current holdings are below the target, buy additional shares,
             subject to available cash.

        Importantly, the target number of shares is adjusted for transaction fees:
          - For selling, the effective revenue per share is reduced by self.sell_cost_pct.
          - For buying, the effective cost per share is increased by self.buy_cost_pct.
          
        We compute two targets:
            target_shares_sell = (desired_allocation * total_value) / (price * (1 - sell_cost_pct))
            target_shares_buy  = (desired_allocation * total_value) / (price * (1 + buy_cost_pct))

        Parameters:
            actions (np.ndarray): A vector of desired portfolio allocation proportions (summing to 1)
                                  for each stock.
        """
        current_prices = self._current_prices()  # Current prices for each stock
        n_stocks = len(current_prices)
        
        # Validate that the actions match the number of stocks.
        assert len(actions) == n_stocks, f"Actions {len(actions)} != stocks {n_stocks}"


        # Check for turbulence: if detected, liquidate the entire portfolio.
        if self.turbulence_bool[self.current_step] != 0:
            self.balance += np.sum(self.positions * current_prices * (1 - self.sell_cost_pct))
            self.positions[:] = 0
            return

        # Calculate total portfolio value (cash + current value of holdings)
        total_value = self.balance + np.sum(self.positions * current_prices)
        
        # Compute target shares accounting for transaction fees:
        # - If selling, we want to hold fewer shares since the revenue per sold share is lower.
        # - If buying, we need to purchase more shares since each share costs more.
        target_shares_sell = (actions * total_value) / (current_prices * (1 - self.sell_cost_pct))
        target_shares_buy  = (actions * total_value) / (current_prices * (1 + self.buy_cost_pct))


        # SELL PHASE: For each stock, if current holdings exceed the selling target, sell the excess.
        for idx in range(n_stocks):
            if self.positions[idx] > target_shares_sell[idx]:
                shares_to_sell = self.positions[idx] - target_shares_sell[idx]
                revenue = shares_to_sell * current_prices[idx] * (1 - self.sell_cost_pct)
                self.positions[idx] -= shares_to_sell
                self.balance += revenue

        # BUY PHASE: For each stock, if current holdings are below the buying target, buy the shortfall.
        for idx in range(n_stocks):
            if self.positions[idx] < target_shares_buy[idx]:
                shares_to_buy = target_shares_buy[idx] - self.positions[idx]
                cost = shares_to_buy * current_prices[idx] * (1 + self.buy_cost_pct)
                # If insufficient balance, buy as many shares as possible.
                if cost > self.balance:
                    shares_to_buy = self.balance / (current_prices[idx] * (1 + self.buy_cost_pct))
                    cost = shares_to_buy * current_prices[idx] * (1 + self.buy_cost_pct)
                if shares_to_buy > 0:
                    self.positions[idx] += shares_to_buy
                    self.balance -= cost

    def _current_prices(self) -> np.ndarray:
        """Accurate price denormalization"""
        normalized_close = self.dataset.groupby('tic').nth(self.current_step)['close'].values
        return (normalized_close * (self.close_maxs - self.close_mins)) + self.close_mins

    def get_feature_names(self) -> List[str]:
        """
        Generates and returns the list of feature names corresponding to the
        observation vector. This covers:
          - Financial features
          - Stock features per ticker (with ticker identifier)
          - Global system features
        """
        financial_names = ['balance', 'step_ratio', 'market_return', 'market_volatility']
        stock_names = []
        # Assuming tickers are ordered (e.g. alphabetically)
        tickers = sorted(self.dataset.tic.unique())
        for tic in tickers:
            for col in self.stock_cols:
                stock_names.append(f"{tic}_{col}")
        system_names = self.system_cols
        return financial_names + stock_names + system_names

