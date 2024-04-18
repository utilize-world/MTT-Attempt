import torch
import os
from .actor_critic import Actor, Critic
import einops
import torch.nn.functional as F


class MAPPO:
    def __init__(self, args):
        self.args = args
        self.train_step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.policy = Actor(args, 0)
        # 创建单个policy_net,这里的0，就是代表第0个agent的actor，由于单个actor，
        # 在homogeneous情况下随机一个都行
        self.q_lr = 1e-3
        self.p_lr = 3e-4
        self.critic = Critic(args).to(self.device)
        self.actor = Actor(args, 0).to(self.device)

        #    以下是超参，还未加入arguments中
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        ############################################
        # self.actor_t = Actor(args, agent_id)
        # self.actor_t.load_state_dict(self.actor.state_dict())
        # self.qf1_t.load_state_dict(self.qf1.state_dict())
        # self.qf2_t.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = torch.optim.Adam((self.critic.parameters()), lr=self.q_lr)
        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()), lr=self.p_lr)


    # 先留着用来加载和存放model参数
    # def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
    #     """
    #     Calculate value function loss.
    #     :param values: (torch.Tensor) value function predictions.
    #     :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
    #     :param return_batch: (torch.Tensor) reward to go returns.
    #     :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.
    #
    #     :return value_loss: (torch.Tensor) value function loss.
    #     """
    #     value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
    #                                                                                     self.clip_param)
    #     error_clipped = return_batch - value_pred_clipped
    #     error_original = return_batch - values
    #     value_loss_original = (error_original**2) / 2  # MSE
    #     value_loss = value_loss_original
    #     value_loss = value_loss.mean()
        # if self._use_popart or self._use_valuenorm:
        #     self.value_normalizer.update(return_batch)
        #     error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
        #     error_original = self.value_normalizer.normalize(return_batch) - values
        # else:
        #     error_clipped = return_batch - value_pred_clipped
        #     error_original = return_batch - values

        # if self._use_huber_loss:
        #     value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        #     value_loss_original = huber_loss(error_original, self.huber_delta)
        # else:
        #     value_loss_clipped = mse_loss(error_clipped)
        #     value_loss_original = mse_loss(error_original)
        #
        # if self._use_clipped_value_loss:
        #     value_loss = torch.max(value_loss_original, value_loss_clipped)
        # else:
        #     value_loss = value_loss_original
        #
        # if self._use_value_active_masks:
        #     value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        # else:
        #     value_loss = value_loss.mean()

        # return value_loss

    def save_model(self, train_step, agent_id):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.policy.state_dict(), model_path + '/' + 'MAPPO' + '/' + num + 'MAPPO_actor_params.pkl')
        torch.save(self.critic.state_dict(), model_path + '/' + 'MAPPO' + '/' + num + 'MAPPO_critic_q1_params.pkl')


    def load_model(self):
        pass

    # def _soft_update_target_network(self):
    #     for target_param, param in zip(self.critic.parameters(), self.qf1_t.parameters()):
    #         target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)
    #
    #
    def train(self, obs, next_obs, dones, actions, rewards, agent_id, time_step):
        """
        PPO训练较为特殊
        transition仍表示为用于训练的数据，但是这里是一个episode中的所有数据
        每个episode更新一次
        """
        rewards = torch.tensor(rewards).to(self.device)
        with torch.no_grad():
            next_value = self.critic(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.args.max_episode_len)):
                if t == self.args.max_episode_len - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
            next_state_log_pi = einops.reduce(
                next_state_log, "c b a -> b a", "sum"
            )
            # 这里相当于把每个agent的log概率加起来，结构应该是这样的
            #
            #                  表示的只是数据的数组形式，并不是一种数据结构
            #   agent\ batch内的sample，logΠ      1       2       3                            1
            #           1                                                          1      prob_sum1
            #           2                                                   ---->  2      prob_sum2
            #           3                                                          3      prob_sum3
            # SAC bellman equation,注意这里的输入是observation的concatenate, 而没有加入global information,这点之后再考虑


            self.q_optimizer.zero_grad()
            qf_loss.backward()
            self.q_optimizer.step()


