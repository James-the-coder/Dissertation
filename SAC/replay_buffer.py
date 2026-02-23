from collections import deque
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HERBuffer():
    def __init__(self, buffer_size=1e6, k_future=4):
        self.buffer = deque([], maxlen=buffer_size)
        self.k_future = k_future


    def store_trajectory(self, trajectory):
        # 1. augment transitions
        augmented_trajectory = self.augment_trajectory(trajectory)

        # 2. push new trajectory to buffer
        for transition in augmented_trajectory:
            self.buffer.append(transition)


    def augment_trajectory(self, trajectory):
        new_trajectory = []
        k_choice = self.k_future
        for idx, transition in enumerate(trajectory):

            new_trajectory.append(transition)
            # (state, achieved_goal, desired_goal, reward)
            future_indices = list(range(idx+1, len(trajectory)))
            if len(future_indices) == 0:
                continue

            k_choice = min(self.k_future, len(future_indices))

            chosen_indices = np.random.choice(future_indices, k_choice, replace=False)
            
            for future_idx in chosen_indices:
                future_obs = trajectory[future_idx]

                augment = transition.copy()

                augment[4] = 0.0 # altering reward
                augment[2] = future_obs[1] # altering goal

                new_trajectory.append(augment)
        return new_trajectory
    

    def sample(self, batch_size):

        indices = np.random.randint(0, len(self.buffer), size=batch_size)

        batch = [self.buffer[i] for i in indices]

        obs, achieved, desired, actions, rewards, next_obs, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(obs)).to(device),
            torch.FloatTensor(np.array(desired)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_obs)).to(device),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)
        )