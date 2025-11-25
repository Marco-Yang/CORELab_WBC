# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# B1Z1 Stairs Climbing Environment (Simple Version)
# Based on ManipLoco, simple stair climbing task inspired by GO2 stair climbing design

import numpy as np
import torch
from typing import Tuple, Dict
from isaacgym.torch_utils import *

from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict

class ManipLocoStairsSimple(ManipLoco):
    """
    B1Z1 stair climbing environment (inspired by GO2 design)
    Features:
    1. Fully based on ManipLoco, keeping all robot configurations unchanged
    2. Only modifies terrain generation to stairs
    3. Incorporates stair climbing reward function design from my_unitree_go2_gym
    4. Adds GO2-style gait control and stability rewards
    5. Reduces arm activity in stair environment, focusing on quadruped climbing
    """
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """
        Initialize B1Z1 stair climbing environment (GO2 design)
        """
        print("Initializing B1Z1 stair climbing environment (GO2-style rewards)...")
        print(f"   Device: {sim_device}")
        print(f"   Headless: {headless}")
        print("   Reward design: Inspired by my_unitree_go2_gym stair climbing")
        
        self.stair_progress_buf = None
        self.current_stair_level = None
        self.stair_climbing_reward_scale = 2.0
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        print("B1Z1 stair environment initialization complete (GO2 reward functions integrated)!")
        
    def _init_buffers(self):
        """Initialize stair-related buffers"""
        super()._init_buffers()
        
        self.stair_progress_buf = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.current_stair_level = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        
        print("Stair-related buffers initialized")

    def _init_height_points(self):
        """Initialize height measurement points"""
        if hasattr(super(), '_init_height_points'):
            return super()._init_height_points()
        
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
        
        self.stair_progress_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.current_stair_level = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        if not hasattr(self, 'feet_air_time'):
            self.feet_air_time = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        if not hasattr(self, 'last_contacts'):
            self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        
        print("Stair-related buffers initialized")
        
    def _reward_stair_climbing(self):
        """Main stair climbing reward - optimized based on B1Z1 config"""
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        tracking_reward = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        
        current_height = self.root_states[:, 2]
        height_progress = torch.clamp(current_height - 0.45, min=0.0, max=1.2)
        height_reward = height_progress * 0.4
        
        forward_vel = torch.clamp(self.base_lin_vel[:, 0], min=0.0, max=0.8)
        command_mask = torch.norm(self.commands[:, :2], dim=1) > 0.05
        forward_reward = forward_vel * command_mask.float()
        
        total_reward = tracking_reward * 0.6 + height_reward * 0.3 + forward_reward * 0.1
        return total_reward
        
    def _reward_stair_stability(self):
        """Stair stability reward - optimized based on B1Z1 config"""
        base_euler = torch.stack(euler_from_quat(self.base_quat), dim=-1)
        orientation_error = torch.sum(torch.square(base_euler[:, :2]), dim=1)
        orientation_reward = torch.exp(-orientation_error * 1.5)
        
        lateral_vel_penalty = torch.abs(self.base_lin_vel[:, 1]) * 0.3
        vertical_vel_penalty = torch.abs(self.base_lin_vel[:, 2]) * 0.4
        velocity_stability = torch.exp(-(lateral_vel_penalty + vertical_vel_penalty))
        
        ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        ang_vel_stability = torch.exp(-ang_vel_xy * 0.15)
        
        stability_reward = orientation_reward * 0.5 + velocity_stability * 0.3 + ang_vel_stability * 0.2
        return stability_reward
        
    def _reward_foot_clearance(self):
        """Foot clearance reward - optimized based on B1Z1 foot control"""
        if not hasattr(self, 'measured_heights') or not torch.is_tensor(self.measured_heights):
            return torch.zeros(self.num_envs, device=self.device)
        
        command_mask = torch.norm(self.commands[:, :2], dim=1) > 0.05
        
        if not command_mask.any():
            return torch.zeros(self.num_envs, device=self.device)
        
        feet_height = self.rigid_body_state[:, self.feet_indices, 2]  
        
        base_z = self.root_states[:, 2]  
        relative_foot_heights = feet_height - base_z.unsqueeze(1)  
        
        target_height = 0.04  
        height_errors = torch.abs(relative_foot_heights - target_height)  # [num_envs, num_feet]
        
        foot_height_reward = torch.mean(height_errors, dim=1)
        
        foot_height_reward = foot_height_reward * command_mask.float()
        
        return foot_height_reward
    
    def _reward_feet_air_time(self):
        """Foot air time reward - GO2 approach based on B1Z1 config"""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, getattr(self, 'last_contacts', contact)) 
        self.last_contacts = contact
        
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        
        target_air_time = 0.25
        rew_airTime = torch.sum((self.feet_air_time - target_air_time) * first_contact, dim=1)
        
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.05
        
        self.feet_air_time *= ~contact_filt
        
        return rew_airTime
    
    def _reward_base_height_stairs(self):
        """
        Stair climbing height reward (using average height of foot contact points)
        - Rewards increase in average height of four foot contact points, not body height
        - Prevents robot from gaining reward by extending legs without climbing stairs
        - Only considers feet actually in contact with ground
        """
        feet_height = self.rigid_body_state[:, self.feet_indices, 2]
        
        foot_contacts = self.foot_contacts_from_sensor
        
        contacted_feet_height = feet_height * foot_contacts.float()
        
        num_contacts = torch.clamp(foot_contacts.sum(dim=1), min=1.0)
        
        avg_contact_height = contacted_feet_height.sum(dim=1) / num_contacts  # [num_envs]
        

        target_height = self.cfg.rewards.base_height_target - 0.38  
        
        height_error = torch.square(avg_contact_height - target_height)
        height_reward = torch.exp(-height_error * 2.5)  
        return height_reward
    
    def _reward_energy_efficiency(self):
        """Energy efficiency reward - optimized based on B1Z1 torque range"""
        leg_torques = self.torques[:, :12]
        torque_penalty = torch.sum(torch.square(leg_torques), dim=1)
        
        if hasattr(self, 'last_actions'):
            action_rate = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        else:
            action_rate = torch.zeros(self.num_envs, device=self.device)
        
        efficiency_reward = torch.exp(-(torque_penalty * 5e-6 + action_rate * 0.005))
        
        return efficiency_reward
        
    def compute_reward(self):
        """Compute rewards including stair-specific components - integrated based on B1Z1 config"""
        super().compute_reward()
        
        if hasattr(self, 'reward_scales'):
            stair_climbing_rew = self._reward_stair_climbing()
            self.rew_buf += stair_climbing_rew * 0.8
            
            stair_stability_rew = self._reward_stair_stability()
            self.rew_buf += stair_stability_rew * 0.4
            
            feet_air_time_rew = self._reward_feet_air_time()
            self.rew_buf += feet_air_time_rew * 0.5
            
            base_height_rew = self._reward_base_height_stairs()
            self.rew_buf += base_height_rew * 0.3
            
            foot_clearance_rew = self._reward_foot_clearance()
            self.rew_buf += foot_clearance_rew * 0.2
            
            energy_rew = self._reward_energy_efficiency()
            self.rew_buf += energy_rew * 0.1
            
            if hasattr(self, 'arm_rew_buf'):
                stair_arm_factor = 0.5  
                self.arm_rew_buf *= stair_arm_factor
        
    def _get_noise_scale_vec(self, cfg):
        return super()._get_noise_scale_vec(cfg)
        
    def step(self, actions):
        """Override step method to call stair-related reward calculation"""
        return super().step(actions)
    
    def reset(self):
        """Override reset method to reset stair-related state and initialize measured_heights"""
        if not hasattr(self, 'measured_heights'):
            if self.cfg.terrain.measure_heights and hasattr(self, 'height_points'):
                self.measured_heights = self._get_heights()
            else:
                num_height_points = getattr(self, 'num_height_points', 187)
                self.measured_heights = torch.zeros(self.num_envs, num_height_points, device=self.device, requires_grad=False)
        
        obs = super().reset()
        return obs
    
    def _post_physics_step_callback(self):
        """Ensure parent callback is called to update measured_heights"""
        super()._post_physics_step_callback()
    
    def post_physics_step(self):
        """Override to ensure debug visualization and measured_heights update are included"""
        if not hasattr(self, 'height_points') and self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        
        if self.cfg.terrain.measure_heights and hasattr(self, 'height_points') and hasattr(self, 'num_height_points'):
            self.measured_heights = self._get_heights()
        elif not hasattr(self, 'measured_heights'):
            num_height_points = getattr(self, 'num_height_points', 187)
            self.measured_heights = torch.zeros(self.num_envs, num_height_points, device=self.device, requires_grad=False)
        
        super().post_physics_step()
        
     
        if ((self.viewer and self.enable_viewer_sync and self.debug_viz) or self.record_video):
            self.gym.clear_lines(self.viewer)
            if hasattr(self, '_draw_ee_goal_curr'):
                self._draw_ee_goal_curr()                       
            if hasattr(self, '_draw_ee_goal_traj'):
                self._draw_ee_goal_traj()                       
            # if hasattr(self, '_draw_collision_bbox'):
            #     self._draw_collision_bbox()
    
    def reset_idx(self, env_ids, **kwargs):
        """
        Partial environment reset (consistent with parent class) + random assignment to different terrain blocks 
        (prevents overfitting). Each reset randomly assigns robots to different terrain blocks to face different 
        stair height combinations, since Isaac Gym's physical mesh cannot be dynamically updated.
        """
        if hasattr(self, 'terrain') and self.cfg.terrain.mesh_type == "trimesh" and len(env_ids) > 0:
            if not hasattr(self, '_reset_counter'):
                self._reset_counter = 0
            
            self._reset_counter += 1
            
            self.terrain_levels[env_ids] = torch.randint(
                0, self.cfg.terrain.num_rows, 
                (len(env_ids),), 
                device=self.device
            )
            self.terrain_types[env_ids] = torch.randint(
                0, self.cfg.terrain.num_cols,
                (len(env_ids),),
                device=self.device
            )
            
            self.env_origins[env_ids] = self.terrain_origins[
                self.terrain_levels[env_ids], 
                self.terrain_types[env_ids]
            ]
            
            if self._reset_counter % 100 == 0:
                if len(env_ids) > 0:
                    print(f"   - Terrain level range: [{self.terrain_levels[env_ids].min().item()}, {self.terrain_levels[env_ids].max().item()}]")
                    print(f"   - Terrain type range: [{self.terrain_types[env_ids].min().item()}, {self.terrain_types[env_ids].max().item()}]")
                unique_blocks = set()
                for i in range(self.num_envs):
                    unique_blocks.add((self.terrain_levels[i].item(), self.terrain_types[i].item()))
        
        super().reset_idx(env_ids, **kwargs)
        
        if hasattr(self, 'stair_progress_buf'):
            self.stair_progress_buf[env_ids] = 0.0
            self.current_stair_level[env_ids] = 0.0