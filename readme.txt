# Stock Trading Reinforcement Learning System

This repository implements a reinforcement learning framework designed to optimize stock portfolio allocation using historical market data and macroeconomic indicators. 

---

## Table of Contents

- [Overview](#overview)
- [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
- [Advanced Actor-Critic Architecture](#advanced-actor-critic-architecture)
  - [Transformer-Based Actor](#transformer-based-actor)
  - [Transformer-Based Critic](#transformer-based-critic)
- [Data Pipeline & Custom Environment](#data-pipeline--custom-environment)
- [Training Process](#training-process)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

The **Stock Trading Reinforcement Learning System** leverages real, high-fidelity financial data to train an agent in allocating capital across multiple stocks. This is achieved by combining robust data processing pipelines with innovative Transformer-based actor and critic networks within a Proximal Policy Optimization (PPO) framework. The system is engineered to capture complex temporal dependencies in market data, enabling more effective portfolio management decisions in a real-world setting.

---

## Proximal Policy Optimization (PPO)

**PPO** is a state-of-the-art policy gradient algorithm widely used in reinforcement learning for its balance between simplicity, sample efficiency, and training stability. Key characteristics of PPO include:

- **Surrogate Objective with Clipping:** PPO maximizes a clipped surrogate objective function that restricts the policy update step—ensuring new policies do not deviate excessively from the current policy.
- **Advantage Estimation:** By computing advantage functions (using Generalized Advantage Estimation, GAE), PPO reduces variance in the policy gradient estimates while still providing low bias.
- **Multiple Epoch Updates:** The algorithm re-uses data from a buffer over several epochs, performing multiple passes over mini-batches to refine both the actor and critic networks.
- **Trust-Region-Like Behavior:** Without the computational overhead of full trust region optimization, PPO effectively constrains policy updates, balancing exploration and exploitation.

This robust framework makes PPO highly appropriate for applications where timely and reliable decision-making is critical—such as navigating the complexities of the stock market.

---

## Advanced Actor-Critic Architecture

Traditional actor and critic networks in reinforcement learning are commonly implemented as simple Multi-Layer Perceptrons (MLPs). In this system, however, both networks are enhanced using transformer architectures to better capture the temporal dynamics inherent in financial time series.

### Transformer-Based Actor

- **Temporal Context Integration:** The actor network augments a standard feedforward network with a transformer encoder to process a sequence of past states. This temporal context is crucial in stock trading where recent trends and patterns inform allocation decisions.
- **Portfolio Allocation via Dirichlet Distribution:** Instead of producing raw action values from an MLP, the actor emits logits that are transformed into probabilities. These probabilities parameterize a Dirichlet distribution, which naturally enforces the constraint that portfolio allocations sum to one.
- **Confidence Control:** A learnable concentration parameter adjusts the variance of the Dirichlet distribution, enabling the model to express varying levels of certainty in its allocation decisions.

### Transformer-Based Critic

- **Sequence Modeling for Future Rewards:** The critic network is designed to predict the expected discounted cumulative reward through a series of simulated future states. It processes sequences of state representations using a temporal transformer encoder, capturing long-range dependencies and market trends.
- **Iterative Forecasting:** Utilizing learnable decoder queries and a transformer decoder, the critic iteratively simulates forward steps into the future. Each step predicts a scalar value which is then discounted and accumulated, furnishing a robust estimate of the state's value.
- **Enhanced Representation:** This transformer-based approach allows the critic to adapt more flexibly to the non-linear, dynamic behavior of financial markets compared to a typical MLP, thereby providing more accurate value estimations for complex time-dependent data.

---

## Data Pipeline & Custom Environment

- **Real Historical Data:** The system collects and processes genuine stock price data along with essential macroeconomic indicators. Data is retrieved from reputable sources like Alpaca (for market prices) and FRED (for economic indicators), ensuring the agent is exposed to realistic market conditions.
- **Feature Engineering:** The data processing pipeline includes advanced steps such as cleaning, normalization, and feature engineering. This involves calculating technical indicators, volatility measures, and macroeconomic derivatives to enrich the input features.
- **Custom Gym Environment:** A tailored Gym environment simulates actual portfolio management under real market conditions. It considers transaction costs, turbulence thresholds, and rebalancing constraints — all calibrated on authentic historical behavior rather than on simulated scenarios.
- **Robust Observations:** The environment constructs a composite state from financial metrics, stock-specific attributes, and global system indicators, providing a comprehensive snapshot necessary for making informed decisions.

---

## Training Process

1. **Data Acquisition & Processing:** The training begins with downloading and processing real market and economic data. The pipeline produces a normalized and feature-rich dataset that feeds the custom Gym environment.
2. **Temporal Sequencing:** To leverage transformer architectures, a sliding window of recent states is maintained. This sequence forms the input context for both the actor and critic networks.
3. **Experience Buffer & Updates:** Transitions are stored in a PPO experience buffer. The agent then performs minibatch gradient updates across multiple epochs using a clipped surrogate objective, applying advantage estimation and handling entropy regularization.
4. **Monitoring with TensorBoard:** The training loop logs detailed metrics including returns, loss functions, gradient norms, and even feature importance analyses. TensorBoard integration provides real-time visualization of these metrics to aid in debugging and model tuning.

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/kyle-curham/TraderBot.git
   cd TraderBot
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *Key Dependencies:*
   - Python >= 3.8
   - PyTorch
   - Pandas
   - NumPy
   - scikit-learn
   - Gym
   - TensorBoard
   - fredapi
   - Matplotlib
   - Seaborn

---

## Configuration

Before running the training script, ensure that you have a `credentials.json` file at the repository root containing API keys and endpoints for data acquisition:

```json
{
  "API_KEY_ALPACA": "your_alpaca_api_key",
  "API_SECRET_ALPACA": "your_alpaca_secret_key",
  "API_BASE_URL_ALPACA": "https://paper-api.alpaca.markets",
  "API_KEY_FRED": "your_fred_api_key"
}
```

Ensure that this file is correctly formatted and includes all required keys.

---

## Usage

To start the training process, simply run:

```bash
python train.py
```

The script will:
- Download and process historical stock and macroeconomic data.
- Initialize the custom trading environment.
- Set up and train the transformer-based PPO agent.
- Launch TensorBoard in a separate thread for real-time training visualization.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch with your improvements or bug fixes.
3. Adhere to PEP 8 guidelines and maintain consistent error handling.
4. Submit a pull request detailing your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgments

- **Alpaca Markets:** For providing reliable stock market data.
- **FRED:** For comprehensive macroeconomic data.
- **Research Community:** For continuous contributions in advancing reinforcement learning algorithms.