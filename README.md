# Intrinsically Motivated Deep RL for Sparse-Reward Robotic Manipulation

Comparing exploration strategies for robotic pick-and-place tasks under sparse rewards, 
by combining SAC+HER with two intrinsic motivation methods (ICM and RND).

📄 [Full dissertation (PDF)](dissertation.pdf) — University of Bath, final year, 2026

## Overview

Sparse rewards make robotic manipulation tasks hard to learn: an agent gets almost no 
signal until it stumbles onto success. This project implements Soft Actor-Critic with 
Hindsight Experience Replay (SAC+HER) from scratch, then augments it with two intrinsic 
motivation modules - Intrinsic Curiosity Module (ICM) and Random Network Distillation (RND) 
to see whether directed exploration improves sample efficiency and whether those policies are
safe for real hardware.

## Method

- **Environment:** MuJoCo robotic pick-and-place tasks (sparse reward)
- **Base algorithm:** SAC + HER, implemented from scratch in Python
- **Exploration bonuses tested:** ICM, RND, and a no-bonus baseline
- **Compute:** Models trained on university compute cluster

## Results

Four SAC+HER variants were evaluated on Fetch-Pick-And-Place-v4 (sparse reward), 
averaged across 4 seeds over 4.75M timesteps each: a fixed-entropy baseline, a 
learnt-entropy baseline, and the learnt-entropy baseline augmented with either RND 
or ICM.
![plots](images/success%20rate%20final.png)
![plots](images/mean%20reward%20final.png)
![plots](images/intrinsic%20reward%20plot.png)
![plots](images/MAJ%20plot%20final.png)

RND performed best overall — 0.93 success rate, converging in 1.3M timesteps, ~200k faster than the strongest baseline — while ICM actually underperformed the learnt-entropy baseline (0.90 success rate, 2.1M timesteps), because MuJoCo's deterministic physics let its forward-dynamics model predict transitions too easily, collapsing its exploration signal early. The more interesting finding came from tracking Mean Absolute Jerk as a proxy for real-world hardware safety: both RND and ICM converged to trajectories with ~250 m/s³ of jerk (peaking over 800 m/s³) — equivalent to 25g/s at the end-effector, well beyond what a physical manipulator could handle without rapid actuator wear — because with fixed-length episodes, both agents solve the task early then "farm" their own exploration bonus through erratic post-task movement. The fixed-entropy baseline, despite the worst success rate, was the only policy safe for zero-shot hardware deployment. The takeaway: the algorithms that win on standard RL benchmarks are the same ones that produce policies unsafe for real robots — success rate alone is a poor proxy for deployment readiness.


## Repo structure
```text
SAC:
  |   actor_net.py
  |   agent_eval.py
  |   config.json
  |   critic_net.py
  |   jitter.py
  |   normaliser.py
  |   parse_config.py
  |   replay_buffer.py
  |   sac_agent.py
  |   sac_agent_hc.py
  |   sac_agent_rnd_cmpre.py
  |   sac_agent_tune_temp.py
  |   
  +---ICM
  |   |   icm.py
  |   |   
  |           
  +---RND
  |   |   rnd.py
  |   |   
  |     
```
## Running it
Adjust the config.json file to reflect the desired hyperparameters
```bash
pip install -r requirements.txt
python SAC/sac_agent.py
```
