import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
from actor_net import Actor

# --- CONFIGURATION ---
ENV_NAME = "FetchPickAndPlace-v4"
#ENV_NAME = "FetchReach-v4"
MODEL_PATH = "./saves v1/sac_her_fetch_final.pth" # Change to a specific checkpoint if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_agent():
    # 1. Setup Environment with rendering
    gym.register_envs(gymnasium_robotics)

    env = gym.make(ENV_NAME, render_mode="rgb_array", max_episode_steps=50)

    # video wrapper
    env = RecordVideo(env, video_folder="./videos v2", name_prefix="FetchPickAndPlace_Demo", episode_trigger=lambda x: True)

    # 2. Get Dimensions to initialize the Actor
    obs_sample, _ = env.reset()
    state_dim = obs_sample['observation'].shape[0]
    goal_dim = obs_sample['desired_goal'].shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 3. Initialize Actor and Load Weights
    print(f"Loading model from {MODEL_PATH}...")
    actor = Actor(state_dim, goal_dim, action_dim, max_action).to(device)
    
    # Load the dictionary and extract just the actor's weights
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    
    # Set to evaluation mode (good practice, disables dropout/batchnorm if you had them)
    actor.eval() 

    # 4. Evaluation Loop
    episodes_to_watch = 20
    
    for ep in range(episodes_to_watch):
        obs_dict, _ = env.reset()
        state = obs_dict['observation']
        desired_goal = obs_dict['desired_goal']
        
        episode_reward = 0
        success = False
        
        for t in range(50):
            #time.sleep(0.1)
            # Format inputs for PyTorch
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            goal_tensor = torch.FloatTensor(desired_goal).unsqueeze(0).to(device)
            
            # Predict action (DETERMINISTIC)
            with torch.no_grad():
                mean, _, _ = actor(state_tensor, goal_tensor)
                # We only take the mean, apply tanh, and scale it
                action = torch.tanh(mean).cpu().numpy()[0] * max_action
                
            # Step environment
            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            
            state = next_obs_dict['observation']
            episode_reward += reward
            
            # Check if it solved the task
            if info.get('is_success', 0.0) > 0:
                success = True
                
            if terminated or truncated:
                break
                
        print(f"Episode {ep+1} | Reward: {episode_reward} | Success: {'YES' if success else 'NO'}")

    env.close()

if __name__ == "__main__":
    evaluate_agent()