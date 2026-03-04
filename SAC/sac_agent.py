from actor_net import Actor
from critic_net import Critic
from replay_buffer import HERBuffer
import torch
import torch.optim as optim
import copy
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
import gymnasium_robotics
import os 
import numpy as np
import pandas as pd
from parse_config import load_settings
from pathlib import Path

script_dir = Path(__file__).parent.absolute()

ENV_NAME = "FetchPickAndPlace-v4"
#ENV_NAME = "FetchReach-v4"
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAC_Agent():  
    def __init__(self, state_dim, goal_dim, action_dim, max_action, gamma=0.98, tau=0.05, alpha=0.2, alpha_fixed=True, learn_rate=3e-4):
        self.actor = Actor(state_dim, goal_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, goal_dim, action_dim).to(device)

        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)  

        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=learn_rate)
        self.critic_optimiser = optim.Adam(self.critic.parameters(), lr=learn_rate)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        
        self.alpha_fixed = alpha_fixed
        if not self.alpha_fixed:
            # alpha is tunable here
            self.target_entropy = -float(action_dim)
            # optimise for log(alpha) better numerical stability
            self.log_alpha = torch.tensor([np.log(alpha)], dtype=torch.float32, requires_grad=True, device=device)
            self.alpha_optimiser = optim.Adam([self.log_alpha], lr=learn_rate)
        else:
            self.alpha = torch.tensor([alpha], dtype=torch.float32, device=device)


    def select_action(self, state, goal, deterministic=False):

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        goal = torch.FloatTensor(goal).unsqueeze(0).to(device)

        with torch.no_grad():
            mean, log_std, _ = self.actor.forward(state, goal)

            if deterministic:
                action = torch.tanh(mean) * self.max_action
            else:
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)

                # reparameterisation
                z = dist.rsample()
                action = torch.tanh(z) * self.max_action

        return action.cpu().numpy()[0]
    
    def train(self, replay_buffer, batch_size=256):

        state, goal, action, reward, next_state, done = replay_buffer.sample(batch_size)

        if self.alpha_fixed:
            alpha_val = self.alpha

        else:
            alpha_val = self.log_alpha.exp().detach()
        
        with torch.no_grad():
            # target action
            next_mean, next_log_std, _ = self.target_actor.forward(next_state, goal)
            next_std = next_log_std.exp()
            next_dist = torch.distributions.Normal(next_mean, next_std)
            next_action_sample = next_dist.sample()
            next_action = torch.tanh(next_action_sample) * self.max_action

            # computing action distribution entropy
            log_prob = next_dist.log_prob(next_action_sample) - torch.log(1 - torch.tanh(next_action_sample).pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)

            # target q value calculation
            target_Q1, target_Q2 = self.target_critic.forward(next_state, goal, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - alpha_val * log_prob
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # critic update
        current_Q1, current_Q2 = self.critic.forward(state, goal, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()

        # actor update
        # re-evaluate current state
        mean, log_std, _ = self.actor.forward(state, goal)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        new_action = torch.tanh(z) * self.max_action

        log_prob = dist.log_prob(z) - torch.log(1 - torch.tanh(z).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        Q1, Q2 = self.critic(state, goal, new_action)
        Q = torch.min(Q1, Q2)

        actor_loss = (alpha_val * log_prob - Q).mean()

        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        if not self.alpha_fixed:
            # calculate alpha loss
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimiser.zero_grad()
            alpha_loss.backward()
            self.alpha_optimiser.step()

            alpha_loss_value = alpha_loss.item()

        # target network polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item(), alpha_val.item(), alpha_loss_value, 

    def save_checkpoint(self, filename):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimiser.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimiser.state_dict(),
        }
        if not self.alpha_fixed:
            checkpoint['log_alpha'] = self.log_alpha
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimiser.state_dict()
            
        torch.save(checkpoint, filename)


    
    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimiser.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimiser.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        if not self.alpha_fixed and 'log_alpha' in checkpoint:
            self.log_alpha.data = checkpoint['log_alpha'].data
            self.alpha_optimiser.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        # Also sync target networks
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

# --- 5. MAIN TRAINING LOOP ---
if __name__ == "__main__":
    config_settings = load_settings(f"{script_dir}/config.json")
    env_name, episodes, sac_settings = config_settings

    alpha_fixed, temp_val, batch_size, buffer_size, k_future, sac_tau, sac_gamma, sac_lr = sac_settings

    print(f"LOADED CONFIG:")
    print(f"ENV: {env_name}")
    print(f"Episodes: {episodes}")
    print(f"SAC: (temp_fixed={alpha_fixed}, batch={batch_size}, buffer={buffer_size},\nk_future={k_future}, tau={sac_tau}, gamma={sac_gamma}, lr={sac_lr})")
    gym.register_envs(gymnasium_robotics)
    env = gym.make(env_name)
    #env = NormalizeObservation(env)

    # Create directories for saving
    if not os.path.exists("./saves"):
        os.makedirs("./saves")
    
    # Dimensions
    obs_sample, _ = env.reset()
    state_dim = obs_sample['observation'].shape[0]
    goal_dim = obs_sample['desired_goal'].shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SAC_Agent(state_dim, goal_dim, action_dim, max_action, gamma=sac_gamma, 
                      tau=sac_tau, alpha=temp_val, alpha_fixed=alpha_fixed, learn_rate=sac_lr)
    replay_buffer = HERBuffer(buffer_size=buffer_size, k_future=k_future)

    # episodes = 2000
    # batch_size = 256

    # --- Logging Setup ---
    training_logs = {
        "episode": [],
        "reward": [],
        "success_rate": [],
        "actor_loss": [],
        "critic_loss": [],
        "alpha": [],
        "alpha_loss": []
    }
    
    print(f"Starting training on {device}...")

    for ep in range(episodes):
        obs_dict, _ = env.reset()
        state = obs_dict['observation']
        desired_goal = obs_dict['desired_goal']
        
        episode_trajectory = []
        episode_reward = 0
        success = 0
        
        # 50 Steps per episode in Fetch envs
        for t in range(50):
            # Select Action
            action = agent.select_action(state, desired_goal, deterministic=False)
            
            # Step
            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            next_state = next_obs_dict['observation']
            next_achieved_goal = next_obs_dict['achieved_goal']
            next_desired_goal = next_obs_dict['desired_goal']

            # Check success (Fetch envs provide 'is_success' in info)
            if info.get('is_success', 0.0) > 0:
                success = 1
            
            done = terminated or truncated
            episode_reward += reward
            
            # --- PACK THE TRANSITION ---
            # Order: [obs, achieved_goal, desired_goal, action, reward, next_obs, done]
            transition = [
                state,                  # 0
                next_achieved_goal,     # 1 (Use 'next' achieved for HER logic)
                desired_goal,           # 2
                action,                 # 3
                reward,                 # 4
                next_state,             # 5
                float(done)             # 6
            ]
            episode_trajectory.append(transition)
            
            state = next_state
            if done:
                break
        
        # End of Episode: Store and Augment
        replay_buffer.store_trajectory(episode_trajectory)

        # Train and collect losses
        actor_losses = []
        critic_losses = []
        alpha_losses = []
        alpha_vals = []
        
        if len(replay_buffer.buffer) > batch_size:
            for _ in range(50):
                # Capture losses returned by agent.train()
                a_loss, c_loss, alpha_v, alpha_loss = agent.train(replay_buffer, batch_size)
                actor_losses.append(a_loss)
                critic_losses.append(c_loss)
                alpha_losses.append(alpha_loss)
                alpha_vals.append(alpha_v)

        
        # Calculate averages for this episode
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0

        if alpha_vals:
            avg_alpha = np.mean(alpha_vals)
            avg_alpha_loss = np.mean(alpha_losses)
        else:
            avg_alpha = agent.alpha.item() if agent.alpha_fixed else agent.log_alpha.exp().item()
            avg_alpha_loss = 0.0

        # --- LOGGING DATA ---
        training_logs["episode"].append(ep + 1)
        training_logs["reward"].append(episode_reward)
        training_logs["success_rate"].append(success)
        training_logs["actor_loss"].append(avg_actor_loss)
        training_logs["critic_loss"].append(avg_critic_loss)
        training_logs["alpha"].append(avg_alpha)
        training_logs["alpha_loss"].append(avg_alpha_loss)


        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}: Total Reward: {episode_reward}")

        # --- SAVE DATA PERIODICALLY ---
        # Save CSV log every 50 episodes
        if (ep + 1) % 50 == 0:
            df = pd.DataFrame(training_logs)
            df.to_csv("./saves/training_log.csv", index=False)
        
        # Save Model Checkpoint every 500 episodes
        if (ep + 1) % 500 == 0:
            agent.save_checkpoint(f"./saves/sac_her_fetch_{ep+1}.pth")
            print(f"Model Saved: sac_her_fetch_{ep+1}.pth")

    # Final Save
    agent.save_checkpoint("./saves/sac_her_fetch_final.pth")
    df = pd.DataFrame(training_logs)
    df.to_csv("./saves/training_log.csv", index=False)
    print("Training Complete. Models and logs saved.")
