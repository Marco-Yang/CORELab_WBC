# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifi    except KeyboardInterrupt:
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import get_load_path

import numpy as np
import torch
import time
import sys

np.set_printoptions(precision=3, suppress=True)

def play(args):
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    
    # Force use flat terrain task but load original b1z1 model
    original_task = args.task
    args.task = "b1z1_flat"
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    
    env_cfg.terrain.num_rows = 6
    env_cfg.terrain.num_cols = 3
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.randomize_base_com = False

    if args.observe_gait_commands:
        env_cfg.env.observe_gait_commands = True

    if args.stand_by:
        env_cfg.env.stand_by = True

    # check if observation matches
    from legged_gym.utils.helpers import update_cfg_from_args
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)
    
    if not hasattr(args, 'headless') or not args.headless:
        args.headless = False
    
    # Prepare environment with flat terrain
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # Restore original task name for model loading
    args.task = original_task
    
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, checkpoint, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    print("Success! Low-level model loaded and running in flat terrain environment!")
    total_params = sum(p.numel() for p in ppo_runner.alg.actor_critic.parameters())
    print(f"Total model parameters: {total_params}")
    print(f"Environment info: {env.cfg.env.num_envs} envs, observation dim: {env.cfg.env.num_observations}")
    
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    
    traj_length = 1000*int(env.max_episode_length)
    
    print(f"Starting simulation for {traj_length} steps...")
    print("Robot will move according to environment commands")
    print("Press Ctrl+C to stop simulation")
    
    env.reset()
    
    import time
    try:
        for i in range(traj_length):
            start_time = time.time()
            
            actions = policy(obs.detach(), hist_encoding=True)
            obs, _, rews, arm_rews, dones, infos = env.step(actions.detach())
            
            if not args.headless:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)

            if args.headless:
                if i % 1000 == 0:
                    print(f"Step: {i:6d}, Total reward: {rews.item():.3f}, Arm reward: {arm_rews.item():.3f}")
            
            stop_time = time.time()
            duration = stop_time - start_time
            time.sleep(max(0.02 - duration, 0))
            
    except KeyboardInterrupt:
        print(f"User interrupted, stopped at step {i}")
    
    print("Simulation ended!")

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)