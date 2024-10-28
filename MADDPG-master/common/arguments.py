import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Training
    parser.add_argument("--training-times", type=int, default=20, help="numbers of Training")
    parser.add_argument("--cuda", type=bool, default=True, help="cuda enable")
    parser.add_argument("--writer", type=bool, default=True, help='enable tensorboard')

    # Environment
    parser.add_argument("--scenario-name", type=str, default="MLGA2C", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=200, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=200000, help="number of time steps")
    # 一个地图最多env.n个agents，用户可以定义min(env.n,num-adversaries)个敌人，剩下的是好的agent
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.2, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    # SAC relevant factor
    parser.add_argument("--alpha", type=float, default=0.05, help="Entropy regularization coefficient")
    parser.add_argument("--autotune", type=bool, default=False, help="automatic tuning of the entropy coefficient")
    parser.add_argument("--update-interval", type=int, default=1, help="regulate the update of policy network")
    # PPO relevant paras
    parser.add_argument("--centralized-input", type=bool, default=True, help="if true, the MAPPO will use centralized critic")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="gae factor")
    parser.add_argument("--update-epi", type=int, default=80, help="update times each ")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="clip para")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="used for training")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")
    parser.add_argument("--fig-save-dir", type=str, default="./figures", help="directory in which save the sns_plot figure")
    parser.add_argument("--csv-save-dir", type=str, default="./data_p", help="directory in which training data(including rewards at which training number)")
    parser.add_argument("--tensorboard-dir", type=str, default='tensorboard_data',help="directory in which stores the data using in tensorboard")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=100, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=200, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=10000, help="how often to evaluate model")
    args = parser.parse_args()

    return args
