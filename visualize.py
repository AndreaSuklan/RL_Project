import gymnasium as gym
from gymnasium import spaces
import pygame
import Box2D
from Box2D import (b2World, b2PolygonShape, b2CircleShape, b2_staticBody, b2_dynamicBody, b2FixtureDef, e_wheelJoint, b2RayCastCallback)
from stable_baselines3 import PPO
import numpy as np
import math
from race import *

class RayCastCallback(b2RayCastCallback):
    def __init__(self, **kwargs):
        b2RayCastCallback.__init__(self)
        self.fixture = None
        self.point = None
        self.normal = None
        self.fraction = 1.0

    def ReportFixture(self, fixture, point, normal, fraction):
        self.fixture = fixture
        self.point = point
        self.normal = normal
        self.fraction = fraction
        return fraction
    
if __name__ == '__main__':
    # assumeing it's in the same directory.
    MODEL_PATH = "ppo_hcr_model.zip"
    NUM_EPISODES = 10


    env = HillClimbEnv(render_mode="human")

    try:
        model = PPO.load(MODEL_PATH, env=env)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please make sure the trained model file exists in the same directory as this script.")
        exit()

    print("Starting Visualization")
    print(f"Model: {MODEL_PATH}")
    print(f"Running for {NUM_EPISODES} episodes.")

    for episode in range(NUM_EPISODES):

        obs, info = env.reset()
        
        print(f"\nStarting Episode {episode + 1}/{NUM_EPISODES}")
        
        while True:

            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                if terminated:
                    print("Episode finished: Terminated (e.g., car flipped).")
                elif truncated:
                    print("Episode finished: Truncated (e.g., time limit reached).")
                break
    
    env.close()
    print("\nVisualization finished")