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
[One or two key plots here — e.g. success rate / sample efficiency vs training steps, 
comparing baseline vs ICM vs RND]

[1-2 sentences on the headline finding — e.g. "RND improved sample efficiency by X% 
over the SAC+HER baseline in the hardest task variant, while ICM underperformed due to..."]

## What I'd do differently

[2-3 honest sentences — e.g. more seeds for statistical significance, a harder task 
suite, comparing against a learned dense reward baseline]

## Repo structure

├── src/           # SAC, HER, ICM, RND implementations
├── configs/       # experiment configs
├── scripts/       # training / eval entry points
├── docs/
│   └── dissertation.pdf
└── results/       # plots, logs

## Running it

\`\`\`bash
pip install -r requirements.txt
python scripts/train.py --config configs/rnd_pickplace.yaml
\`\`\`

## Skills demonstrated
Python · PyTorch · Reinforcement Learning · MuJoCo · Experiment design · 
Multi-node GPU training
