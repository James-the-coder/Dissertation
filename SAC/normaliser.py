import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    def __init__(self, input_dim, epsilon=1e-4):
        super(RunningMeanStd, self).__init__()
        self.epsilon = epsilon

        self.register_buffer('mean', torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer('var', torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer('count', torch.zeros(1, dtype=torch.int32))

    def update(self, x):
        """Updates the running statistics given a batch of observations."""
        x = x.to(torch.float32, device=self.mean.device)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)


    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Welford's online algorithm for batch updates."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        # Update mean
        new_mean = self.mean + delta * batch_count / tot_count
        
        # Update variance
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


    def normalise(self, x, clip_val=5.0):
        """Normalises and clips the input."""
        x = x.to(torch.float32)
        # Using epsilon to avoid division by zero
        x_norm = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        return torch.clamp(x_norm, min=-clip_val, max=clip_val)
