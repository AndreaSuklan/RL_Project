# ppo.py

from stable_baselines3 import PPO

def create_agent(env, log_dir):
    """
    Creates and returns a PPO agent.
    This function currently uses the Stable-Baselines3 library,
    but you can replace it with your own PPO implementation later.
    
    Args:
        env: The Gymnasium environment.
        log_dir: The directory to save TensorBoard logs.
        
    Returns:
        A PPO model instance.
    """
    print("Creating PPO agent...")
    agent = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir
    )
    return agent