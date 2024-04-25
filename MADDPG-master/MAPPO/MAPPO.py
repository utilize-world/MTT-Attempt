import torch
import os
from .actor_critic import Actor, Critic
import einops
import torch.nn.functional as F
import numpy as np

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
        # self.clip_param = args.clip_param
        # self.ppo_epoch = args.ppo_epoch
        # self.num_mini_batch = args.num_mini_batch
        # self.data_chunk_length = args.data_chunk_length
        # self.value_loss_coef = args.value_loss_coef
        # self.entropy_coef = args.entropy_coef
        # self.max_grad_norm = args.max_grad_norm
        # self.huber_delta = args.huber_delta
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
        model_path = os.path.join(model_path, "MAPPO")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.policy.state_dict(), model_path + '/' + num + 'MAPPO_actor_params.pkl')
        torch.save(self.critic.state_dict(), model_path + '/' + num + 'MAPPO_critic_q1_params.pkl')


    def load_model(self):
        pass

    # def _soft_update_target_network(self):
    #     for target_param, param in zip(self.critic.parameters(), self.qf1_t.parameters()):
    #         target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)
    #
    #
    def train(self, obs, next_obs, values, dones, actions, logprobs, rewards, nextdone, agent_id, time_steps):
        """
        PPO训练较为特殊
        从obs到rewards仍表示为用于训练的数据，但是这里是一个episode中的所有数据
        每个episode更新一次
        b_inds, agent_id, time_steps
        分别是
        """
        rewards = torch.tensor(rewards).to(self.device)
        next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(nextdone).to(self.device)
        next_obs = [i for i in next_obs]
        next_obs = torch.cat(next_obs, dim=2)   # 处理一下next_obs
        with torch.no_grad():
            next_value = self.critic(next_obs.squeeze(1)).reshape(-1)    # 处理一下维度，从200*1*38变成200*38
            advantages = torch.zeros_like(rewards).to(self.device).reshape(-1)
            lastgaelam = 0
            for t in reversed(range(self.args.max_episode_len)):
                if t == self.args.max_episode_len - 1:
                    nextnonterminal = 1.0 - next_done[t]
                    nextvalues = next_value[t]
                    ######################## next_value[t] maybe mistake!!!!!!!!!!!!
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values.reshape(-1)

        # flatten
        # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_obs = obs[agent_id].reshape((-1, self.args.obs_shape[agent_id]))
        obs_joint = [i for i in obs]
        obs_joint = torch.cat(obs_joint, dim=2)  # 处理一下next_obs
        b_obs_joint = obs_joint.reshape((-1,  sum(self.args.obs_shape)))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,  self.args.action_shape[agent_id]))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_inds = np.arange(time_steps)
        clipfracs = []
        #np.random.shuffle(b_inds)
        for epoch in range(self.args.update_epi):
            np.random.shuffle(b_inds)
            min_size = int(time_steps/10) + 1
            for start in range(0, time_steps, min_size):
                end = start + min_size
                mb_inds = b_inds[start:end]

                #_, newlogprob, entropy, newvalue = agent.get_action_and_value(obs[mb_inds], actions.long()[mb_inds])
                _, newlogprob, entropy = self.actor.get_actions(b_obs[mb_inds], b_actions.long()[mb_inds])
                newvalue = self.critic(b_obs_joint[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if True:    # use normalized adv func
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if True: # toggles whether or not to use a clipped loss for the value function, as per the paper
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                break

        # y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        # var_y = np.var(y_true)
        # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y



