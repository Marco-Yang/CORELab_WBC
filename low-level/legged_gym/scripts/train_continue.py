# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
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

import numpy as np
import os
from datetime import datetime
import isaacgym

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import wandb

def train_continue(args):
    """
    Continue training function - resume training from existing weights.
    """
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    try:
        os.makedirs(log_pth)
    except:
        pass
    
    if args.debug:
        mode = "disabled"
        args.rows = 6
        args.cols = 2
        args.num_envs = 128
    else:
        mode = "online"
    
    wandb.init(project=args.proj_name, name=args.exptid, mode=mode, dir=LEGGED_GYM_ENVS_DIR + "/logs")
    
    if args.task == "b1z1_stairs_new":
        wandb.save(LEGGED_GYM_ENVS_DIR + "/manip_loco/b1z1_stairs_new_config.py", policy="now")
        wandb.save(LEGGED_GYM_ENVS_DIR + "/manip_loco/manip_loco_stairs_simple.py", policy="now")
    else:
        wandb.save(LEGGED_GYM_ENVS_DIR + "/manip_loco/b1z1_config.py", policy="now")
        wandb.save(LEGGED_GYM_ENVS_DIR + "/manip_loco/manip_loco.py", policy="now")

    print(f"Creating environment for task: {args.task}")
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    print(f"Creating algorithm runner...")

    ppo_runner, train_cfg, current_checkpoint = task_registry.make_alg_runner(
        log_root=log_pth, 
        env=env, 
        name=args.task, 
        args=args
    )
    
    if args.pretrained_path:
        print(f"Loading pretrained weights from: {args.pretrained_path}")
        if not os.path.exists(args.pretrained_path):
            raise FileNotFoundError(f"Pretrained model not found at: {args.pretrained_path}")
        
        loaded_dict = torch.load(args.pretrained_path, map_location=ppo_runner.device)
        
        if 'model_state_dict' in loaded_dict:
            ppo_runner.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
            print("Successfully loaded actor_critic weights from pretrained model")
        else:
            print("Warning: 'model_state_dict' not found in pretrained model")
        
        if args.load_optimizer and 'optimizer_state_dict' in loaded_dict:
            ppo_runner.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            print("Successfully loaded optimizer state from pretrained model")
        else:
            print("Using fresh optimizer state for new training")
        
        if args.start_iteration:
            ppo_runner.set_it(args.start_iteration)
            print(f"Starting training from iteration: {args.start_iteration}")
        else:
            model_filename = os.path.basename(args.pretrained_path)
            if "model_" in model_filename:
                try:
                    iteration_num = int(model_filename.split("_")[-1].split(".")[0])
                    ppo_runner.set_it(iteration_num)
                    print(f"Auto-detected starting iteration: {iteration_num}")
                except:
                    print("Could not auto-detect iteration number, starting from 0")
        
        if args.reset_noise_std:
            init_std = train_cfg.policy.init_noise_std if hasattr(train_cfg.policy, 'init_noise_std') else 1.0
            ppo_runner.alg.actor_critic.reset_std(init_std, env.num_actions, device=ppo_runner.device)
            print(f"Reset action noise std to: {init_std}")
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"   Task: {args.task}")
    print(f"   Experiment ID: {args.exptid}")
    print(f"   Project Name: {args.proj_name}")
    print(f"   Number of Environments: {env.num_envs}")
    print(f"   Max Iterations: {train_cfg.runner.max_iterations}")
    print(f"   Device: {args.rl_device}")
    if args.pretrained_path:
        print(f"   Pretrained Model: {args.pretrained_path}")
    print(f"{'='*60}\n")
    
    print("Starting continued training...")
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations, 
        init_at_random_ep_len=True
    )
    
    print("Training completed successfully!")

def get_continue_args():
    """Get command line arguments for continue training."""
    custom_parameters = [
        {"name": "--task", "type": str, "default": "b1z1_stairs_new", "help": "Task name for continued training"},
        {"name": "--pretrained_path", "type": str, "required": True, "help": "Path to pretrained model (e.g., logs/b1z1-low/model_38000/model_38000.pt)"},
        {"name": "--load_optimizer", "action": "store_true", "default": False, "help": "Load optimizer state from pretrained model"},
        {"name": "--reset_noise_std", "action": "store_true", "default": True, "help": "Reset action noise std for new task exploration"},
        {"name": "--start_iteration", "type": int, "help": "Starting iteration number (auto-detected if not specified)"},
        
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str, "help": "Name of the experiment to run or load"},
        {"name": "--run_name", "type": str, "required": False, "help": "Name of the run"},
        {"name": "--load_run", "type": str, "default": "", "help": "Name of the run to load when resume=True"},
        {"name": "--checkpoint", "type": int, "default": "-1", "help": "Saved model checkpoint number"},
        {"name": "--stop_update_goal", "action": "store_true", "help": "Stop when update a new ee goal"},
        {"name": "--observe_gait_commands", "action": "store_true", "help": "Observe gait commands"},
        
        {"name": "--exptid", "type": str, "required": True, "help": "Experiment ID for continued training"},
        {"name": "--debug", "action": "store_true", "default": False, "help": "Disable wandb logging"},
        {"name": "--proj_name", "type": str, "default": "b1z1-stairs-continued", "help": "Project name for continued training"},
        {"name": "--resumeid", "type": str, "help": "Resume experiment ID"},

        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "Device used by the RL algorithm"},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create"},
        {"name": "--seed", "type": int, "help": "Random seed"},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations"},
        {"name": "--stochastic", "action": "store_true", "default": False, "help": "Use stochastic actions"},
        {"name": "--use_jit", "action": "store_true", "default": False, "help": "Use jit"},
        {"name": "--record_video", "action": "store_true", "default": False, "help": "Record video"},
        {"name": "--stand_by", "action": "store_true", "default": False, "help": "Stand by"},
        {"name": "--flat_terrain", "action": "store_true", "default": False, "help": "Flat terrain"},
        {"name": "--pitch_control", "action": "store_true", "default": False, "help": "Control Pitch"},
        {"name": "--vel_obs", "action": "store_true", "default": False, "help": "Velocity observations"},
        
        {"name": "--rows", "type": int, "help": "Number of terrain rows"},
        {"name": "--cols", "type": int, "help": "Number of terrain cols"},
    ]
    
    import argparse
    from isaacgym import gymutil
    
    args = gymutil.parse_arguments(
        description="Continued RL Policy Training",
        custom_parameters=custom_parameters)
    
    args.test = False
    
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == 'cuda':
        args.sim_device += f":{args.sim_device_id}"
    
    return args

if __name__ == '__main__':
    args = get_continue_args()
    
    print(f"\n{'='*80}")
    print(f" B1Z1 CONTINUED TRAINING - STAIRS CLIMBING")
    print(f"{'='*80}")
    print(f" Arguments:")
    print(f"   Task: {args.task}")
    print(f"   Pretrained Path: {args.pretrained_path}")
    print(f"   Experiment ID: {args.exptid}")
    print(f"   Project Name: {args.proj_name}")
    print(f"   Device: {args.rl_device}")
    print(f"   Load Optimizer: {args.load_optimizer}")
    print(f"   Reset Noise Std: {args.reset_noise_std}")
    if args.start_iteration:
        print(f"   Start Iteration: {args.start_iteration}")
    print(f"{'='*80}\n")
    
    train_continue(args)