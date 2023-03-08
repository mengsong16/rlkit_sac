from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from ssac.utils.config_utils import parse_config
from path import *  
from ssac.envs.nav_2d.point_env import env_load_fn
import gym
from rlkit.envs.wrappers import ProxyEnv

class PointEnvWrapper(ProxyEnv):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space['observation']


    def reset(self, **kwargs):
        obs_dict = self._wrapped_env.reset(**kwargs)
        return obs_dict['observation']

    def step(self, action):
        obs_dict, reward, done, info = self._wrapped_env.step(action)
        return obs_dict['observation'], reward, done, info

    
def experiment(variant):
    expl_env = PointEnvWrapper(env_load_fn(environment_name=variant.get("train_environment_name"),
                config=variant))
    eval_env = PointEnvWrapper(env_load_fn(environment_name=variant.get("train_environment_name"),
                config=variant))
    
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    #config = parse_config(os.path.join(config_path, "half_cheetach.yaml"))
    #config = parse_config(os.path.join(config_path, "continuous_sac_hallway.yaml"))
    #config = parse_config(os.path.join(config_path, "continuous_sac_empty.yaml"))
    config = parse_config(os.path.join(config_path, "continuous_sac_corridor.yaml"))
    
    setup_logger(exp_prefix=config.get("experiment_prefix"), 
                 variant=config,
                 seed=config.get("seed"),
                 base_log_dir=runs_path,
                 snapshot_mode="gap_and_last",
                 snapshot_gap=config.get("save_every_epochs"),
                 )
    ptu.set_gpu_mode(True)
    experiment(config)
