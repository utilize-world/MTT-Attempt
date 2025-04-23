# MTT-Attempt

This is a pytorch implementation for multi-target tracking tasks in tailored environment based on [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs), the corresponding paper of MADDPG is [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275).

The proposed MTT-SN adopts MADDPG algorithm with designed safe nets, which can address the issues that tracking while avoiding collisions in MTT tasks. The baselines are MADDPG with reward shaping (MADDPG-RS), MASAC-RS, and MASAC-SN. 

## Requirement

- python=3.8.1
- torch=2.1.0

## Note

+ we are still under development...
