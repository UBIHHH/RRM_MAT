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
from mat.config import get_config
from mat.envs.rrm.RRM_env import RRMEnv
from mat.runner.shared.rrm_runner import RRMRunner
from mat.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from mat.envs.env_wrappers import ShareDummyVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "RRM":
                env = RRMEnv(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    if all_args.n_rollout_threads == 1:
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
    parser.add_argument('--n_mbs', type=int, default=0, help="number of macro base stations")
    parser.add_argument('--n_pbs', type=int, default=3, help="number of pico base stations")
    parser.add_argument('--n_fbs', type=int, default=0, help="number of femto base stations")
    parser.add_argument('--n_ues', type=int, default=10, help="number of user equipments")
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
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = 3  # 3 base stations

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    from mat.algorithms.mat.mat_trainer import MATTrainer as TrainerSingle
    from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as PolicySingle
    from mat.utils.shared_buffer import SharedReplayBuffer

    if all_args.share_policy:
        policy = PolicySingle(all_args, 
                             envs.observation_space[0], 
                             envs.share_observation_space[0],
                             envs.action_space[0], 
                             num_agents, 
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