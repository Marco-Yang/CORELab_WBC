# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# B1Z1 Ring Slope Environment

import numpy as np
import torch
from typing import Tuple, Dict
from isaacgym.torch_utils import *

from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict

class ManipLocoRingSlope(ManipLoco):
    """B1Z1 ring slope climbing environment based on ManipLoco"""
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        print("Initializing B1Z1 ring slope climbing environment...")
        print(f"   Device: {sim_device}")
        print(f"   Headless: {headless}")
        
        self.slope_progress_buf = None
        self.current_slope_level = None
        self.slope_climbing_reward_scale = 2.0
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        print("B1Z1 ring slope environment initialized!")
        
    def _init_buffers(self):
        super()._init_buffers()
        
        self.slope_progress_buf = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.current_slope_level = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        
        print("Slope buffers initialized")

    def _init_height_points(self):
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
        

        self.slope_progress_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.current_slope_level = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        

        if not hasattr(self, 'feet_air_time'):
            self.feet_air_time = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        if not hasattr(self, 'last_contacts'):
            self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        
        print("Slope buffers initialized")

    def _reward_slope_climbing(self):
        base_height = self.root_states[:, 2]
        height_progress = torch.clamp(base_height - 0.45, min=0.0, max=1.5)
        height_reward = height_progress * 0.4
        
        center_distance = torch.norm(self.root_states[:, :2], dim=1)
        distance_reward = torch.clamp(center_distance / 10.0, min=0.0, max=0.8)
        
        forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        robot_heading = quat_apply_yaw(self.base_quat, forward_vec)
        
        center_to_robot = self.root_states[:, :2] / (center_distance.unsqueeze(1) + 1e-8)

        direction_alignment = torch.sum(robot_heading[:, :2] * center_to_robot, dim=1)
        direction_reward = torch.clamp(direction_alignment, min=0.0, max=1.0) * 0.3
        
        total_reward = height_reward * 0.5 + distance_reward * 0.3 + direction_reward * 0.2
        return total_reward

    def _reward_slope_stability(self):
        base_euler = torch.stack(euler_from_quat(self.base_quat), dim=-1)
        roll_error = torch.square(base_euler[:, 0])
        pitch_error = torch.square(torch.clamp(torch.abs(base_euler[:, 1]) - 0.3, min=0.0))
        orientation_error = roll_error + pitch_error
        orientation_reward = torch.exp(-orientation_error * 1.2)
        
        lateral_vel_penalty = torch.abs(self.base_lin_vel[:, 1]) * 0.2
        vel_stability = torch.exp(-lateral_vel_penalty)
 
        ang_vel_xy = torch.norm(self.base_ang_vel[:, :2], dim=1)
        ang_vel_penalty = torch.clamp(ang_vel_xy - 0.5, min=0.0) * 0.4
        ang_vel_stability = torch.exp(-ang_vel_penalty)
        
        stability_reward = orientation_reward * 0.5 + vel_stability * 0.3 + ang_vel_stability * 0.2
        return stability_reward

    def _reward_foot_clearance(self):
        feet_heights = self.rigid_body_state[:, self.feet_indices, 2]
        base_height = self.root_states[:, 2].unsqueeze(1)
        
        relative_feet_heights = feet_heights - base_height + 0.35

        target_clearance = 0.2
        clearance_error = torch.abs(relative_feet_heights - target_clearance)
        clearance_reward = torch.exp(-clearance_error * 3.0).mean(dim=1)
        
        return clearance_reward

    def _reward_feet_air_time(self):
        contact_thresh = 5.0
        contact = self.contact_forces[:, self.feet_indices, 2] > contact_thresh
        
        if not hasattr(self, 'feet_air_time'):
            self.feet_air_time = torch.zeros_like(contact, dtype=torch.float)
            self.last_contacts = torch.zeros_like(contact, dtype=torch.bool)
        
        self.feet_air_time += self.dt
        
        first_contact = (self.feet_air_time > 0.) * contact * ~self.last_contacts
        
        optimal_air_time = 0.4
        air_time_quality = torch.exp(-torch.abs(self.feet_air_time - optimal_air_time) / 0.15)
        rew_air_time = torch.sum(air_time_quality * first_contact, dim=1)
        
        self.feet_air_time *= ~contact
        self.last_contacts = contact
        
        return rew_air_time

    def _reward_base_height_slopes(self):
        """Reward base height increase for upward climbing"""
        base_height = self.root_states[:, 2]
        
        min_height = 0.4
        max_height = 1.5
        target_height = 0.8
        
        if hasattr(self, 'initial_height'):
            height_progress = base_height - self.initial_height
        else:
            height_progress = base_height - min_height
            
        height_error = torch.abs(base_height - target_height)
        height_reward = torch.exp(-height_error * 2.5)
        
        return height_reward

    def compute_reward(self):
        super().compute_reward()
        
        if hasattr(self, 'reward_scales'):
            slope_climbing_rew = self._reward_slope_climbing()
            self.rew_buf += slope_climbing_rew * 0.8
            
            slope_stability_rew = self._reward_slope_stability()
            self.rew_buf += slope_stability_rew * 0.4
            
            feet_air_time_rew = self._reward_feet_air_time()
            self.rew_buf += feet_air_time_rew * 0.5
            
            base_height_rew = self._reward_base_height_slopes()
            self.rew_buf += base_height_rew * 0.3
            
            foot_clearance_rew = self._reward_foot_clearance()
            self.rew_buf += foot_clearance_rew * 0.2

    def _get_noise_scale_vec(self, cfg):
        return super()._get_noise_scale_vec(cfg)
        
    def step(self, actions):
        return super().step(actions)
        
    def reset(self):
        if not hasattr(self, 'measured_heights'):
            if self.cfg.terrain.measure_heights and hasattr(self, 'height_points'):
                self.measured_heights = self._get_heights()
            else:
                num_height_points = getattr(self, 'num_height_points', 187)
                self.measured_heights = torch.zeros(self.num_envs, num_height_points, device=self.device, requires_grad=False)
        
        obs = super().reset()
        
        if hasattr(self, 'root_states'):
            self.initial_height = self.root_states[:, 2].clone()
        
        return obs

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        
    def post_physics_step(self):
        super().post_physics_step()
        
    def reset_idx(self, env_ids, **kwargs):
        """Reset with randomized terrain assignment to prevent overfitting"""
        
        if hasattr(self, 'terrain') and self.cfg.terrain.mesh_type == "trimesh" and len(env_ids) > 0:
            if not hasattr(self, '_reset_counter'):
                self._reset_counter = 0
            
            self._reset_counter += 1
            
            # Randomly assign terrain levels (slope angles from 3° to 30°)
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
                    print(f" Ring slope reset #{self._reset_counter}:")
                    print(f"   - Slope angle levels: [{self.terrain_levels[env_ids].min().item()}, {self.terrain_levels[env_ids].max().item()}] (0=3°, 9=30°)")
                    print(f"   - Terrain types: [{self.terrain_types[env_ids].min().item()}, {self.terrain_types[env_ids].max().item()}]")
                unique_blocks = set()
                for i in range(self.num_envs):
                    unique_blocks.add((self.terrain_levels[i].item(), self.terrain_types[i].item()))
                if self._reset_counter % 100 == 0:
                    print(f"   - Using {len(unique_blocks)} unique slope terrain blocks")
        
        super().reset_idx(env_ids, **kwargs)