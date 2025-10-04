import numpy as np
import gym
# from RRM import Environment
import numpy as np
from marlcustomeEnv import MarlCustomEnv4
from types import SimpleNamespace
from gym.spaces import Box

class RRMEnv:
    def __init__(self, args):
        self.args = args

        self.sce = self._create_scenario_from_args(args)

        self.env = MarlCustomEnv4(self.sce)
        self.num_agents = self.env.BS_num
        self.n_agents = self.num_agents  # ShareDummyVecEnv 需要

        # 每个 agent 的真实动作维度
        self._agent_act_dims = [space.shape[0] for space in self.env.action_spaces]
        # 统一到最大维度
        self._max_act_dim = max(self._agent_act_dims)
        padded_action_space = Box(low=0.0,
                                  high=1.0,
                                  shape=(self._max_act_dim,),
                                  dtype=np.float32)
        self.action_space = [padded_action_space] * self.num_agents
        # obs 取每个 BS 的 len(UE_set)*nRB，pad 到最大值
        # 1) 记录每个 agent 的真实 obs 维度，以及最大维度
        ue_counts = [len(bs.UE_set) for bs in self.env.BSs]
        self.local_obs_dims = [count * self.sce.nRBs for count in ue_counts]
        
        # 使用固定的最大观测空间维度，确保环境实例间的一致性
        # 理论上每个BS最多可以关联的UE数量基于覆盖范围，实际中可能超过配置的Nb
        # 为了保险起见，使用所有UE数量作为上限
        theoretical_max_ues_per_bs = self.sce.nUEs  # 最极端情况：所有UE都在一个BS范围内
        self.max_local_obs_dim = theoretical_max_ues_per_bs * self.sce.nRBs

        # 2) 用 max_local_obs_dim 构造统一的 observation_space
        padded_obs_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_local_obs_dim,),
            dtype=np.float32
        )
        self.observation_space = [padded_obs_space] * self.num_agents

        # 3) 同理记录共享 obs 的维度
        share_dim = self.num_agents * self.sce.nUEs * self.sce.nRBs
        self.share_obs_dim = share_dim
        full_share_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(share_dim,),
            dtype=np.float32
        )
        self.share_observation_space = [full_share_space] * self.num_agents
        print("max_local_obs_dim:", self.max_local_obs_dim)


        self.step_count = 0

    def _create_scenario_from_args(self, args):
        sce = SimpleNamespace(
            nUEs = args.n_ues,
            nRBs = args.n_rbs,
            nMBS = args.n_mbs,
            nPBS = args.n_pbs,
            nFBS = args.n_fbs,
            rMBS = args.r_mbs,
            rPBS = args.r_pbs,
            rFBS = args.r_fbs,
            txpowerMBSdBm = args.txpower_mbs_dbm,
            txpowerPBSdBm = args.txpower_pbs_dbm,
            txpowerFBSdBm = args.txpower_fbs_dbm,
            BW = args.bandwidth,
            N0 = args.noise_power,
            QoS_thr = args.qos_thr,
            fc = args.fc,
            x_max = args.x_max,
            y_max = args.y_max,
            Nb = args.nb,
            Nrb = args.nrb_max,
            bsloclist = None,
            prt = False,
        )
        return sce

    def reset(self, seed=None, options=None):
        self.step_count = 0
        obs_flat, _ = self.env.reset(seed=seed, options=options)
        
        if self.env.history_channel_information is None:
            self.env.history_channel_information = np.zeros(
                (self.num_agents, self.sce.nUEs, self.sce.nRBs), 
                dtype=np.float32
            )
            dummy_action = np.zeros(sum(self._agent_act_dims))
            self.env.get_obs_4baseline(dummy_action)
        
        obs_list = self._get_obs()
        share_obs_list = self._get_share_obs(obs_list)
        
        obs = np.stack(obs_list, axis=0)[None, ...]
        share_obs = np.stack(share_obs_list, axis=0)[None, ...]
        available_actions = None
        return obs, share_obs, available_actions

    def step(self, actions):
        flat_act = None
        if isinstance(actions, np.ndarray):
            if actions.ndim == 3:
                acts = actions[0]               # (n_agents, max_act_dim)
            elif actions.ndim == 2:
                acts = actions                  # (n_agents, max_act_dim)
            elif actions.ndim == 1:
                # 已经是一维扁平动作，但需要检查是否是真实维度
                flat_act = actions
            else:
                raise ValueError(f"Unexpected action ndim: {actions.ndim}")
        else:
            # list of arrays，先拼成 (n_agents, max_act_dim)
            acts = np.stack(actions, axis=0)

        if flat_act is None:
            # 从 padded actions 中提取真实维度的动作
            real_agent_actions = [
                acts[i, : self._agent_act_dims[i]]
                for i in range(self.num_agents)
            ]
            
            # 分离每个agent的UA和RB动作，然后分别拼接
            all_ua_actions = []
            all_rb_actions = []
            
            for idx, agent_action in enumerate(real_agent_actions):
                # 计算每个agent的UA动作维度
                ua_dim = self._agent_act_dims[idx] // (self.env.sce.nRBs + 1)
                
                # 分离UA和RB动作
                ua_part = agent_action[:ua_dim]
                rb_part = agent_action[ua_dim:]
                
                all_ua_actions.append(ua_part)
                all_rb_actions.append(rb_part)
            
            # [所有UA] + [所有RB]
            flat_act = np.concatenate([
                np.concatenate(all_ua_actions),  # 所有agent的UA动作
                np.concatenate(all_rb_actions)   # 所有agent的RB动作
            ])
            
        
        # print(f"[raw flat_act] min={flat_act.min():.4f}, max={flat_act.max():.4f}")

        # 再做映射
        normed = self._normalize_actions(flat_act)
        # 如果动作不在 [0, 1] 范围内，报错
        if np.any(normed < 0.0) or np.any(normed > 1.0):
            raise ValueError(f"Normalized actions out of bounds: {normed}")
        flat_act = normed
        # todo 映射回0-1再给到reward计算
        
        # def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        #     """
        #     Rescale the action from [-1, 1] to [low, high]
        #     (no need for symmetric action space)

        #     :param scaled_action: Action to un-scale
        #     """
        #     assert isinstance(
        #         self.action_space, spaces.Box
        #     ), f"Trying to unscale an action using an action space that is not a Box(): {self.action_space}"
        #     low, high = self.action_space.low, self.action_space.high
        #     return low + (0.5 * (scaled_action + 1.0) * (high - low))
        
        # 问问gpt 如何将1d tensor里的数值映射到给定范围【a,b]之间
        # min-max-normalization； gussian normalization；

        # 检查你这里的计算的数值和 MarlCustomEnv4 中的 MARLstep_withCurrentH 是否一致
        # 注意：这里的 flat_act 需要是一个一维数组，包含所有 agent 的动作
        # 删除那个翻转符号的

        # 验证 flat_act 的维度是否正确
        expected_total_dim = sum(self._agent_act_dims)
        if len(flat_act) != expected_total_dim:
            print(f"Warning: flat_act dimension mismatch. Expected: {expected_total_dim}, Got: {len(flat_act)}")
            print(f"Agent action dims: {self._agent_act_dims}")
            
            # 如果维度不匹配，截断或填充
            if len(flat_act) > expected_total_dim:
                flat_act = flat_act[:expected_total_dim]
            else:
                padded_act = np.zeros(expected_total_dim)
                padded_act[:len(flat_act)] = flat_act
                flat_act = padded_act

        rewards_agent = self._calculate_rewards(flat_act)

        obs_list       = self._get_obs()
        share_obs_list = self._get_share_obs(obs_list)
        obs       = np.stack(obs_list,       axis=0)[None, ...]
        share_obs = np.stack(share_obs_list, axis=0)[None, ...]
        rewards   = rewards_agent.reshape(self.num_agents)[None, :, None]
        self.step_count += 1
        done_flag   = (self.step_count >= self.args.episode_length)
        dones       = np.array([[[done_flag] for _ in range(self.num_agents)]])
        infos       = [{} for _ in range(self.num_agents)]
        available_actions = None

        return obs, share_obs, rewards, dones, infos, available_actions
    
    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        将动作映射到[0,1]区间
        
        :param actions: 原始动作数组
        :return: 映射到[0,1]区间的动作数组
        """
        return self.map_to_unit_interval_sigmoid(actions)

    def map_to_unit_interval_sigmoid(self, real_values: np.ndarray) -> np.ndarray:
        """
        使用Sigmoid函数将实数映射到(0,1)区间
        
        :param real_values: 任意实数数组
        :return: 映射到(0,1)区间的数组
        """
        # 添加数值稳定性检查
        clipped_values = np.clip(real_values, -500, 500)  # 防止exp溢出
        return 1.0 / (1.0 + np.exp(-clipped_values))

    def map_to_unit_interval_tanh(self, real_values: np.ndarray) -> np.ndarray:
        """
        使用tanh函数将实数映射到[0,1]区间
        
        :param real_values: 任意实数数组
        :return: 映射到[0,1]区间的数组
        """
        return 0.5 * (np.tanh(real_values) + 1.0)
    
    def _get_obs(self) -> list[np.ndarray]:
        """
        从 MarlCustomEnv4.history_channel_information 中读取上次 step/reset
        保存的 channel_power 矩阵（shape=(BS_num * nUEs * nRBs,) 或 (1, …)），
        reshape 成 (BS_num, nUEs, nRBs)，然后对每个 BS b：
          - 只取它 own b.UE_set 对应的全局 UE slice
          - flatten 成一维
        最后 pad 到所有 agent 的最大长度。
        """
        # 确保 history_channel_information 已经过 reset/step 填充
        if self.env.history_channel_information is None:
            obs0, _ = self.env.reset()
            self.env.history_channel_information = obs0.copy()
        # flatten + reshape
        flat = np.array(self.env.history_channel_information).ravel()
        BS_num = self.num_agents
        nUEs = self.env.sce.nUEs
        nRBs = self.env.sce.nRBs
        h3 = flat.reshape(BS_num, nUEs, nRBs)
        # 按每个 BS 的 UE_set 拆分
        obs_list = []
        for b_idx, bs in enumerate(self.env.BSs):
            # UE_set 里是 global UE 索引，从 1 开始减 1
            idx = np.array(bs.UE_set, dtype=int) - 1
            local = h3[b_idx, idx, :]      # shape = (len(idx), nRBs)
            obs_list.append(local.ravel())
        # pad 到统一长度
        padded = []
        for o in obs_list:
            if o.size < self.max_local_obs_dim:
                buf = np.zeros(self.max_local_obs_dim, dtype=o.dtype)
                buf[: o.size] = o
                padded.append(buf)
            else:
                padded.append(o)
        return padded

    def _get_share_obs(self, obs_list: list[np.ndarray]) -> list[np.ndarray]:
        """
        这里直接广播每个 agent 能看到全部 BS 的 channel info：
        share_obs_list[i] = 全量 H_flatten
        """
        flat_all = np.array(self.env.history_channel_information).ravel()
        assert flat_all.size == self.share_obs_dim
        return [flat_all.copy() for _ in range(self.num_agents)]
    
    def _calculate_rewards(self, flat_act: np.ndarray) -> np.ndarray:
        obs, reward, terminated, truncated, info = self.env.MARLstep_withCurrentH(flat_act)        
        return np.full(self.num_agents, reward, dtype=np.float32)
            

    
    def seed(self, seed=None):
        np.random.seed(seed)
    
    def close(self):
        pass