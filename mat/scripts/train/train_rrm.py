#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
sys.path.append("../../")
sys.path.append("/home/laughtale/RRM_MAT")
from mat.config import get_config
from mat.envs.rrm.RRM_env import RRMEnv
from mat.runner.shared.rrm_runner import RRMRunner
from mat.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from mat.envs.env_wrappers import ShareDummyVecEnv
import yaml


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "RRM":
                env = RRMEnv(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            # 不同环境不同种子
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    if all_args.n_rollout_threads == 1:#单进程和多进程
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "RRM":
                env = RRMEnv(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument('--config_env', type=str,
                        default='config_environment_setting_MAT.yaml',
                        help="path to yaml env config")

    parser.add_argument('--n_mbs', type=int, default=0, help="number of macro base stations")
    parser.add_argument('--n_pbs', type=int, default=5, help="number of pico base stations")
    parser.add_argument('--n_fbs', type=int, default=0, help="number of femto base stations")
    parser.add_argument('--n_ues', type=int, default=50, help="number of user equipments")
    parser.add_argument('--n_channels', type=int, default=5, help="number of channels")
    parser.add_argument('--r_mbs', type=float, default=500, help="radius of macro base station")
    parser.add_argument('--r_pbs', type=float, default=300, help="radius of pico base station")
    parser.add_argument('--r_fbs', type=float, default=100, help="radius of femto base station")
    parser.add_argument('--txpower_mbs_dbm', type=float, default=43, help="transmit power of macro base station in dBm")
    parser.add_argument('--txpower_pbs_dbm', type=float, default=36, help="transmit power of pico base station in dBm")
    parser.add_argument('--txpower_fbs_dbm', type=float, default=23, help="transmit power of femto base station in dBm")
    parser.add_argument('--bandwidth', type=float, default=180e3, help="channel bandwidth in Hz")
    parser.add_argument('--noise_power', type=float, default=-174, help="noise power in dBm/Hz")
    parser.add_argument('--frequency', type=float, default=2.5, help="carrier frequency in GHz")
    parser.add_argument('--x_max', type=float, default=1000, help="maximum x coordinate")
    parser.add_argument('--y_max', type=float, default=1000, help="maximum y coordinate")
    parser.add_argument('--print_config', action='store_true', default=False, help="whether to print environment configuration")
    parser.add_argument('--obs_dim', type=int, default=30, help="dimension of observation space")
    
    all_args = parser.parse_known_args(args)[0]
    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    all_args.use_eval = True
    print(f"Forced use_eval to: {all_args.use_eval}")

    # 1) load env config
    with open(all_args.config_env, 'r') as f:
        env_cfg = yaml.safe_load(f)

    # 打印 RRM 环境配置
    if all_args.print_config:
        print("===== RRM 环境配置 =====")
        for key, val in env_cfg.items():
            print(f"{key}: {val}")
        print("========================")

    # 2) override all_args
    all_args.n_ues           = env_cfg['nUEs']
    all_args.n_rbs           = env_cfg['nRBs']
    all_args.n_mbs           = env_cfg['nMBS']
    all_args.n_pbs           = env_cfg['nPBS']
    all_args.n_fbs           = env_cfg['nFBS']
    all_args.r_mbs           = env_cfg['rMBS']
    all_args.r_pbs           = env_cfg['rPBS']
    all_args.r_fbs           = env_cfg['rFBS']
    all_args.txpower_mbs_dbm = env_cfg['txpowerMBSdBm']
    all_args.txpower_pbs_dbm = env_cfg['txpowerPBSdBm']
    all_args.txpower_fbs_dbm = env_cfg['txpowerFBSdBm']
    all_args.bandwidth       = env_cfg['BW']
    all_args.n_channel       = env_cfg['nChannel']
    all_args.noise_power     = env_cfg['N0']
    all_args.qos_thr         = env_cfg['QoS_thr']
    all_args.fc              = env_cfg['fc']
    all_args.x_max           = env_cfg['x_max']
    all_args.y_max           = env_cfg['y_max']
    all_args.nb              = env_cfg['Nb']
    all_args.nrb_max         = env_cfg['Nrb']

    # 3) 重算基站数量 & agents
    all_args.num_agents = (all_args.n_mbs
                           + all_args.n_pbs
                           + all_args.n_fbs)

    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mat" or all_args.algorithm_name == "mat_dec":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True

    if all_args.cuda and torch.cuda.is_available():
        print("Using CUDA...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Using CPU...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" + str(all_args.seed),
            group=all_args.scenario_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True
        )
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
                               str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    # 提取第一个子环境的 local_obs_dims
    if hasattr(envs, "envs"):
        local_obs_dims = envs.envs[0].max_local_obs_dim
    else:
        # 如果是 SubprocVecEnv，则：
        local_obs_dims = envs.venv.envs[0].max_local_obs_dim

    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    # num_agents = 3  # 3 base stations

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir
    }

    from mat.algorithms.mat.mat_trainer import MATTrainer as TrainerSingle
    from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as PolicySingle
    from mat.utils.shared_buffer import SharedReplayBuffer

    all_args.num_agents = (all_args.n_mbs
                           + all_args.n_pbs
                           + all_args.n_fbs)

    # ← 在这里定义局部 num_agents 变量
    num_agents = all_args.num_agents

    # 策略网络
    if all_args.share_policy:
        policy = PolicySingle(all_args, 
                             envs.observation_space[0], 
                             envs.share_observation_space[0],
                             envs.action_space[0], 
                             num_agents,
                             local_obs_dims,
                             device)
    else:
        raise NotImplementedError

    buffer = SharedReplayBuffer(
        all_args,
        num_agents,
        envs.observation_space[0],
        envs.share_observation_space[0]
            if all_args.use_centralized_V else envs.observation_space[0],
        envs.action_space[0],
        all_args.env_name
    )

    trainer = TrainerSingle(
        all_args,
        policy,
        num_agents,
        device=device
    )

    from tensorboardX import SummaryWriter
    
    if all_args.use_wandb:
        writter = SummaryWriter(str(run_dir))
    else:
        writter = SummaryWriter(str(run_dir))
        
    config["buffer"] = buffer
    config["trainer"] = trainer
    config["writter"] = writter
    config["log_dir"] = run_dir
    
    runner = RRMRunner(config)
    
    runner.run()
    
    envs.close()
    if all_args.use_eval:
        eval_envs.close()
        
    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()

if __name__ == "__main__":
    main(sys.argv[1:])