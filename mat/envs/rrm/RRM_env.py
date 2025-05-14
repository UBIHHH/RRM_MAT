import numpy as np
from gymnasium.spaces import Box, Discrete
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from RRM import Environment, BS, UE

class RRMEnv:
    def __init__(self, args):
        self.args = args
        self.sce = self._create_scenario_from_args(args)
        self.env = Environment(self.sce)
        
        self.num_agents = self.env.BS_num
        self.n_agents = self.num_agents
        
        self.action_space = []
        for _ in range(self.num_agents):
            self.action_space.append(Discrete(args.n_channels))
        
        self.observation_space = []
        for _ in range(self.num_agents):
            self.observation_space.append(Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32))
        
        self.share_observation_space = [Box(low=-np.inf, high=np.inf, shape=(obs_dim * self.num_agents,), dtype=np.float32) for _ in range(self.num_agents)]
    
    def _create_scenario_from_args(self, args):
        class Scenario:
            def __init__(self, args):
                self.nMBS = args.n_mbs if hasattr(args, 'n_mbs') else 0
                self.nPBS = args.n_pbs if hasattr(args, 'n_pbs') else 3
                self.nFBS = args.n_fbs if hasattr(args, 'n_fbs') else 0
                self.nUEs = args.n_ues if hasattr(args, 'n_ues') else 10
                self.nChannel = args.n_channels if hasattr(args, 'n_channels') else 5
                self.rMBS = args.r_mbs if hasattr(args, 'r_mbs') else 500
                self.rPBS = args.r_pbs if hasattr(args, 'r_pbs') else 300
                self.rFBS = args.r_fbs if hasattr(args, 'r_fbs') else 100
                self.txpowerMBSdBm = args.txpower_mbs_dbm if hasattr(args, 'txpower_mbs_dbm') else 43
                self.txpowerPBSdBm = args.txpower_pbs_dbm if hasattr(args, 'txpower_pbs_dbm') else 36
                self.txpowerFBSdBm = args.txpower_fbs_dbm if hasattr(args, 'txpower_fbs_dbm') else 23
                self.BW = args.bandwidth if hasattr(args, 'bandwidth') else 180e3  # 180 kHz
                self.N0 = args.noise_power if hasattr(args, 'noise_power') else -174  # dBm/Hz
                self.fc = args.frequency if hasattr(args, 'frequency') else 2.5  # GHz
                self.x_max = args.x_max if hasattr(args, 'x_max') else 1000
                self.y_max = args.y_max if hasattr(args, 'y_max') else 1000
                self.bsloclist = args.bs_locations if hasattr(args, 'bs_locations') else None
                self.prt = args.print_config if hasattr(args, 'print_config') else False
        
        return Scenario(args)
    
    def reset(self):
        self.step_count = 0
        self.env.random_walk()
        obs = self._get_obs()
        share_obs = self._get_share_obs(obs)
        available_actions = None
        return obs, share_obs, available_actions
        
        obs = self._get_obs()
        share_obs = self._get_share_obs(obs)
        
        available_actions = None
        
        return obs, share_obs, available_actions
    
    def step(self, actions):
        self.step_count += 1

        rewards = self._calculate_rewards(actions)
        next_obs = self._get_obs()
        next_share_obs = self._get_share_obs(next_obs)

        done = (self.step_count >= self.args.episode_length)
        dones = np.array([[done] for _ in range(self.num_agents)], dtype=bool)

        infos = [{} for _ in range(self.num_agents)]
        available_actions = None

        if done:
            self.step_count = 0

        return next_obs, next_share_obs, rewards, dones, infos, available_actions
    
    def _get_obs(self):
        obs = []
        for i, bs in enumerate(self.env.BSs):
            bs_obs = np.zeros(self.args.obs_dim, dtype=np.float32)
            
            obs_idx = 0
            for ue_id in bs.UE_set:
                if obs_idx + 3 > self.args.obs_dim:
                    break
                    
                ue = self.env.UEs[ue_id-1]
                
                loc_diff = np.array(bs.BS_Loc) - np.array(ue.location)
                distance = np.sqrt(np.sum(loc_diff**2))
                
                try:
                    rx_power, h_power = self.env.test_cal_Receive_Power(bs, distance)
                except AttributeError:
                    rx_power = 0.0
                    h_power = 0.0
                
                bs_obs[obs_idx] = distance
                bs_obs[obs_idx+1] = rx_power
                bs_obs[obs_idx+2] = h_power
                
                obs_idx += 3
            
            obs.append(bs_obs)
        
        return obs

    def _get_share_obs(self, obs):
        share_obs = []
        flat_obs = np.concatenate(obs)
        for _ in range(self.num_agents):
            share_obs.append(flat_obs.copy())
        return share_obs
    
    def _calculate_rewards(self, actions):
        rewards = np.zeros((self.num_agents, 1), dtype=np.float32)
    
        channel_assignment = {}
        
        for bs_idx, action in enumerate(actions):
            bs = self.env.BSs[bs_idx]
            channel = action.item()
            
            if channel not in channel_assignment:
                channel_assignment[channel] = []
            
            for ue_id in bs.UE_set:
                ue = self.env.UEs[ue_id-1]
                channel_assignment[channel].append((bs_idx, ue_id-1))
        
        for channel, assignments in channel_assignment.items():
            for bs_idx, ue_idx in assignments:
                bs = self.env.BSs[bs_idx]
                ue = self.env.UEs[ue_idx]
                
                loc_diff = np.array(bs.BS_Loc) - np.array(ue.location)
                distance = np.sqrt(np.sum(loc_diff**2))
                signal_power, _ = self.env.test_cal_Receive_Power(bs, distance)
                
                interference = 0
                for other_bs_idx, other_ue_idx in assignments:
                    if other_bs_idx != bs_idx:
                        other_bs = self.env.BSs[other_bs_idx]
                        loc_diff = np.array(other_bs.BS_Loc) - np.array(ue.location)
                        distance = np.sqrt(np.sum(loc_diff**2))
                        intf_power, _ = self.env.test_cal_Receive_Power(other_bs, distance)
                        interference += intf_power
                
                noise_power = 10 ** (self.sce.N0 / 10) * self.sce.BW
                
                if interference + noise_power > 0:
                    sinr = signal_power / (interference + noise_power)
                else:
                    sinr = signal_power / 1e-10
                
                data_rate = self.sce.BW * np.log2(1 + sinr) / 1e6  # Mbps
                
                rewards[bs_idx] += data_rate
        
        return rewards
    
    def seed(self, seed=None):
        np.random.seed(seed)
    
    def close(self):
        pass