# MTT-Attempt

This is a pytorch implementation for multi-target tracking tasks in tailored environment based on [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs), the corresponding paper of MADDPG is [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275).

The proposed MTT-SN adopts MADDPG algorithm with designed safe nets, which can address the issues that tracking while avoiding collisions in MTT tasks. The baselines are MADDPG with reward shaping (MADDPG-RS), MASAC-RS, and MASAC-SN. 

## Requirement

- python=3.8.1
- torch=2.3.1

## Note

+ All safety module implementations have been included in `MADDPG-master//SafeModule`. Among the files, `collect_data.py` will collect training batch for constraint_nets, and `train_constraint_networks.py` is then employed to train the constraint_nets. It is noted that before you running the `main.py` with centralized filter, the constraint_nets should be trained in advance.
+ To enable the centralized safe filter, the configuration is in `runner.py`. The `Runner` class with properties includes `non-filter` and `safeNet`. If `non-filter` is `false`, the centralized filter is enabled. More importantly, if safeNet is set to `True`, the `non-filter` must be set to `false`, as the training of safenet depends on the filter. Besides, the `safeNet` and `distill` properties in `MADDPG/MADDPG.py` or `MASAC/MASAC.py` should be set to `True` to enable safe net training.
+ The safe net training should begins after the MARL policy is trained to stable.
+ After configurations are all set, running the `main.py` to begin the training. Or modify the properties `-evaluate` to `True` in `common//arguments.py` to start evaluation. The training data and test data is collected in `data`. And the model is stored in `model` directory.
  
