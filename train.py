import torch
from configs.tickers import DOW_30_TICKER
from data.processor import TradingDataProcessor
from env.trading_env import StockTradingEnv
from agent.ppo_agent import PPOAgent
import pandas as pd  # Added for dtype checking
from tensorboard import program
import webbrowser
import threading


def launch_tensorboard(log_dir: str) -> None:
    """
    Launch TensorBoard for the provided run-specific log directory on host 127.0.0.1,
    open the TensorBoard URL in the default web browser.
    """
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir, '--host', '127.0.0.1'])
    url = tb.launch()
    webbrowser.open(url)
    print("TensorBoard started at:", url)

def train():
    # Initialize components
    time_interval = "1D"
    processor = TradingDataProcessor(base_frequency=time_interval)  # For hourly data
    raw_data = processor.download_data(DOW_30_TICKER, "2020-01-01", "2025-01-01", time_interval)
    processed_data = processor.process_data(raw_data)
    
    # Add data validation statistics
    print("\nProcessed Data Statistics:")
    print("{:<30} {:<10} {:<10} {:<10}".format("Feature", "Min", "Max", "Std"))
    for col in processed_data.columns:
        if col in ['timestamp', 'tic']:
            continue
        if pd.api.types.is_numeric_dtype(processed_data[col]):
            stats = processed_data[col].agg(['min', 'max', 'std']).round(4)
            print("{:<30} {:<10} {:<10} {:<10}".format(
                col, 
                stats['min'], 
                stats['max'], 
                stats['std']
            ))
        else:
            print(f"{col:<30} [Non-numeric column]")

    # Initialize environment and agent
    gamma = 0.99
    env = StockTradingEnv(processed_data, 
                          forecast_horizon=10, 
                          gamma=gamma
                          )

    agent = PPOAgent(
        env=env,
        dataset=processed_data,
        gamma=gamma
    )
    # Launch TensorBoard concurrently using the agent's current log directory.
    tb_thread = threading.Thread(target=launch_tensorboard, args=(agent.writer.log_dir,), daemon=True)
    tb_thread.start()
    
    # Training loop
    for episode in range(1000):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:         
            # Get action and value estimate
            squashed_action, log_prob, value = agent.get_action(state)
            next_state, reward, done, info = env.step(squashed_action)
            port_value = info['raw_value']
            

            # Store transition in buffer
            agent.buffer.store(
                state, 
                squashed_action,
                reward,
                next_state,
                done,
                value,
                log_prob
            )
            
            state = next_state
            
            # Accumulate rewards
            episode_reward += reward
            
            # Update agent at the end of the episode
            if done:
                actor_loss, critic_loss, entropy = agent.update()
                break

        # Calculate percentage change
        initial_balance = 1e4  # Match environment's initial balance
        pct_change = ((port_value - initial_balance) / initial_balance) * 100

        
        # Print episode summary      
        print(f"Episode {episode} | Return: {episode_reward:.6f} | "
              f"Portfolio: {pct_change:.2f}% | "
              f"Actor Loss: {actor_loss:.6f} | Critic Loss: {critic_loss:.6f} | "
              f"Entropy: {entropy:.6f}")
        

        # Log episode summary to TensorBoard
        agent.writer.add_scalar("Episode/Return", episode_reward, episode)
        agent.writer.add_scalar("Episode/PortfolioChange (%)", pct_change, episode)
        agent.writer.add_scalar("Episode/ActorLoss", actor_loss, episode)
        agent.writer.add_scalar("Episode/CriticLoss", critic_loss, episode)
        agent.writer.add_scalar("Episode/Entropy", entropy, episode)
    
    agent.close()

if __name__ == "__main__":
    train()