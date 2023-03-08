from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
import numpy as np
from rlkit.core import logger

from ssac.utils.config_utils import parse_config
from path import *  
from ssac.envs.nav_2d.point_env import env_load_fn
import gym
from rlkit.envs.wrappers import ProxyEnv
from train import PointEnvWrapper  # need this to load environment from data
from ssac.utils.stats import create_stats_ordered_dict
from ssac.envs.nav_2d.tools import plot_trajectories
from ssac.envs.nav_2d.instances import WALLS
import shutil

def save_and_plot(config, env, logs, trajectories, stocastic_eval):
    # get save folder
    save_folder = os.path.join(evaluation_path, 
                            config.get("experiment_prefix"),
                            config.get("eval_experiment_folder"))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # get file name
    checkpoint_name = os.path.splitext(config.get("eval_checkpoint_file"))[0]
    if stocastic_eval:
        stocastic_str = "stocastic"
    else:
        stocastic_str = "deterministic"
    txt_name =  f"{checkpoint_name}-{stocastic_str}-eval_results.txt"
    
    # write file
    with open(os.path.join(save_folder, txt_name), 'w') as outfile:
        for key, value in logs.items():
            outfile.write("%s: %.3f\n"%(key, value))

    print("Saved evaluation file: %s"%(txt_name)) 

    # plot trajectories for 2D nav
    environment_name = config.get("test_environment_name")
    if environment_name in WALLS:
        png_name = f"{checkpoint_name}-{stocastic_str}-trajectories.png"
        save_file_path = os.path.join(save_folder, png_name)

        plot_trajectories(env, trajectories, 
                            config.get("normalize_observation"),
                            save_file_path)
    
        print("Saved plot image: %s"%(png_name)) 


def simulate_policy(config):
    data_path = os.path.join(runs_path, 
                             config.get("experiment_prefix"), 
                             config.get("eval_experiment_folder"), 
                             config.get["eval_checkpoint_file"]) #"params.pkl"
    
    data = torch.load(data_path)

    print("Data loaded: %s"%(data_path))
    
    
    stocastic_eval = config.get("stocastic_eval")
    if stocastic_eval:
        policy = data['exploration/policy']
        policy_prefix = "Stochastic "
    else:
        policy = data['evaluation/policy']
        policy_prefix = "Deterministic "
    
    print(policy_prefix + "policy loaded")

    env = data['evaluation/env']
    print("Environment loaded")
    
    # set gpu
    set_gpu_mode(True)
    policy.cuda()

    num_eval_episodes = int(config.get("num_eval_episodes"))
    max_path_length = int(config["algorithm_kwargs"].get("max_path_length"))

    logs = {}
    trajectories = []
    returns = []
    success_rate = []
    path_lengths = []
    dangerous_times = []

    for episode in list(range(num_eval_episodes)):
        path = rollout(
            env,
            policy,
            max_path_length=max_path_length,
            render=False,
        )

        current_return = np.sum(path['rewards'])
        returns.append(current_return)
        
        last_observation = np.expand_dims(path['next_observations'][-1], axis=0)
        current_trajectory = np.concatenate((path['observations'], last_observation), axis=0)
        trajectories.append(current_trajectory)
        
        num_step = path['rewards'].shape[0]
        path_lengths.append(num_step)
        last_info = path['env_infos'][-1]

        if 'is_success' in last_info.keys():
            success_rate.append(float(last_info['is_success']))
        if 'is_dangerous' in last_info.keys():
            current_dangerous_times = 0
            for info in path['env_infos']:
                current_dangerous_times += float(info['is_dangerous'])
            
            dangerous_times.append(current_dangerous_times)

        
        print('Episode {} finished after {} timesteps.'.format(episode, num_step))
        print('-----------------------------')
    
    logs.update(create_stats_ordered_dict(
            "return",
            returns,
            always_show_all_stats=True,
            exclude_max_min=False,
            exclude_std=False
        ))

    logs.update(create_stats_ordered_dict(
        "path_length",
        path_lengths,
        always_show_all_stats=True,
        exclude_max_min=False,
        exclude_std=False
    ))

    if len(success_rate) > 0:
        logs.update(create_stats_ordered_dict(
            "success_rate",
            success_rate,
            always_show_all_stats=True,
            exclude_max_min=False,
            exclude_std=False
        ))
    
    if len(dangerous_times) > 0:
        logs.update(create_stats_ordered_dict(
            "dangerous_times",
            dangerous_times,
            always_show_all_stats=True,
            exclude_max_min=False,
            exclude_std=False
        ))

    print("================= Evaluation Summary ================")
    for key, value in logs.items():
        print("%s: %.3f"%(key, value))
    print("=====================================================")

    # save and plot
    save_and_plot(config, env, logs, trajectories, stocastic_eval)

if __name__ == "__main__":
    #config = parse_config(os.path.join(config_path, "continuous_sac_hallway.yaml"))
    #config = parse_config(os.path.join(config_path, "continuous_sac_empty.yaml"))
    config = parse_config(os.path.join(config_path, "continuous_sac_corridor.yaml"))
    
    simulate_policy(config)
