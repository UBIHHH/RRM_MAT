import time
import wandb
import numpy as np
from functools import reduce
import torch
import torch.nn as nn
from mat.utils.util import get_gard_norm, huber_loss, mse_loss
from mat.utils.valuenorm import ValueNorm
from mat.algorithms.utils.util import check

def _t2n(x):
    return x.detach().cpu().numpy()

class RRMRunner:
    def __init__(self, config):
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.step_count = 0
        
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        self.model_dir = self.all_args.model_dir

        if 'use_valuenorm' in self.all_args:
            self.use_valuenorm = self.all_args.use_valuenorm
        else:
            self.use_valuenorm = False
        
        self.use_proper_time_limits = self.all_args.use_proper_time_limits
        self.use_max_grad_norm = self.all_args.use_max_grad_norm
        self.max_grad_norm = self.all_args.max_grad_norm
        self.clip_param = self.all_args.clip_param
        self.ppo_epoch = self.all_args.ppo_epoch
        self.num_mini_batch = self.all_args.num_mini_batch
        self.data_chunk_length = self.all_args.data_chunk_length
        self.value_loss_coef = self.all_args.value_loss_coef
        self.entropy_coef = self.all_args.entropy_coef
        self.use_value_active_masks = self.all_args.use_value_active_masks
        self.use_policy_active_masks = self.all_args.use_policy_active_masks
        self.huber_delta = self.all_args.huber_delta
        
        self.trainer = config['trainer']
        
        self.buffer = config['buffer']
        
        self.writter = config['writter']
        self.log_dir = config['log_dir']
        
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        done_episodes_rewards = []

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                    
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic
                
                self.insert(data)

                for i, reward in enumerate(rewards):
                    train_episode_rewards[i] += reward[0]
                    if dones[i][0]:
                        done_episodes_rewards.append(train_episode_rewards[i])
                        train_episode_rewards[i] = 0

            self.compute()
            train_infos = self.train()
            
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
             
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.env_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    print("Average episode rewards is {}.".format(aver_episode_rewards))
                    if self.use_wandb:
                        wandb.log({"average_episode_rewards": aver_episode_rewards}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("average_episode_rewards", {"average_episode_rewards": aver_episode_rewards}, total_num_steps)
                    done_episodes_rewards = []

                self.log_train(train_infos, total_num_steps)

            # Evaluate
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        obs, share_obs, available_actions = self.envs.reset()

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        if available_actions is not None:
            self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            np.concatenate(self.buffer.available_actions[step]) if self.buffer.available_actions is not None else None)
        
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=(1, 2))

        if np.any(dones_env == True):
            rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env, :, :] = 0

        active_masks = (~dones).astype(np.float32)

        bad_masks = np.array([[[1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(
            share_obs, obs, rnn_states, rnn_states_critic,
            actions, action_log_probs, values, rewards, masks, 
            bad_masks, active_masks, available_actions
        )

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()

            if isinstance(eval_available_actions, np.ndarray) and eval_available_actions.ndim > 1:
                avail_arg = np.concatenate(eval_available_actions)
            else:
                avail_arg = None

            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(
                    np.concatenate(eval_share_obs),
                    np.concatenate(eval_obs),
                    np.concatenate(eval_rnn_states),
                    np.concatenate(eval_masks),
                    avail_arg,
                    deterministic=True
                )
            
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=(1, 2))
            
            if np.any(eval_dones_env):
                eval_rnn_states[eval_dones_env] = np.zeros(((eval_dones_env).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env, :, :] = 0

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0)[eval_i])
                    one_episode_rewards = []

            if eval_episode >= self.all_args.eval_episodes:
                eval_env_infos = {
                    'eval_average_episode_rewards': eval_episode_rewards,
                    'eval_max_episode_rewards': [np.max(eval_episode_rewards)]
                }
                
                self.log_env(eval_env_infos, total_num_steps)
                print("Evaluation: average episode rewards: {}.".format(np.mean(eval_episode_rewards)))
                break

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def compute(self):
        self.trainer.prep_rollout()
        cent_obs = np.concatenate(self.buffer.share_obs[-1])
        obs = np.concatenate(self.buffer.obs[-1])
        rnn_states_critic = np.concatenate(self.buffer.rnn_states_critic[-1])
        masks = np.concatenate(self.buffer.masks[-1])
        if self.buffer.available_actions is not None:
            avail = np.concatenate(self.buffer.available_actions[-1])
            next_values = self.trainer.policy.get_values(cent_obs, obs, rnn_states_critic, masks, avail)
        else:
            next_values = self.trainer.policy.get_values(cent_obs, obs, rnn_states_critic, masks)

        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self, episode):
        self.trainer.policy.save(self.log_dir, episode)