import torch
from isaacgym.torch_utils import *

class ManipLoco_rewards:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env
    # -------------Z1: Reward functions----------------

    def _reward_tracking_ee_sphere(self):
        ee_pos_local = quat_rotate_inverse(self.env.base_yaw_quat, self.env.ee_pos - self.env.get_ee_goal_spherical_center())
        ee_pos_error = torch.sum(torch.abs(cart2sphere(ee_pos_local) - self.env.curr_ee_goal_sphere) * self.env.sphere_error_scale, dim=1)
        return torch.exp(-ee_pos_error/self.env.cfg.rewards.tracking_ee_sigma), ee_pos_error

    def _reward_tracking_ee_world(self):
        ee_pos_error = torch.sum(torch.abs(self.env.ee_pos - self.env.curr_ee_goal_cart_world), dim=1)
        rew = torch.exp(-ee_pos_error/self.env.cfg.rewards.tracking_ee_sigma * 2)
        return rew, ee_pos_error

    def _reward_tracking_ee_sphere_walking(self):
        reward, metric = self.env._reward_tracking_ee_sphere()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_tracking_ee_sphere_standing(self):
        reward, metric = self.env._reward_tracking_ee_sphere()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_tracking_ee_cart(self):
        target_ee = self.env.get_ee_goal_spherical_center() + quat_apply(self.env.base_yaw_quat, self.env.curr_ee_goal_cart)
        ee_pos_error = torch.sum(torch.abs(self.env.ee_pos - target_ee), dim=1)
        return torch.exp(-ee_pos_error/self.env.cfg.rewards.tracking_ee_sigma), ee_pos_error
    
    def _reward_tracking_ee_orn(self):
        ee_orn_euler = torch.stack(euler_from_quat(self.env.ee_orn), dim=-1)
        orn_err = torch.sum(torch.abs(torch_wrap_to_pi_minuspi(self.env.ee_goal_orn_euler - ee_orn_euler)) * self.env.orn_error_scale, dim=1)
        return torch.exp(-orn_err/self.env.cfg.rewards.tracking_ee_sigma), orn_err

    def _reward_arm_energy_abs_sum(self):
        energy = torch.sum(torch.abs(self.env.torques[:, 12:-self.env.cfg.env.num_gripper_joints] * self.env.dof_vel[:, 12:-self.env.cfg.env.num_gripper_joints]), dim = 1)
        return energy, energy

    def _reward_tracking_ee_orn_ry(self):
        ee_orn_euler = torch.stack(euler_from_quat(self.env.ee_orn), dim=-1)
        orn_err = torch.sum(torch.abs((torch_wrap_to_pi_minuspi(self.env.ee_goal_orn_euler - ee_orn_euler) * self.env.orn_error_scale)[:, [0, 2]]), dim=1)
        return torch.exp(-orn_err/self.env.cfg.rewards.tracking_ee_sigma), orn_err

    # -------------B1: Reward functions----------------

    def _reward_hip_action_l2(self):
        action_l2 = torch.sum(self.env.actions[:, [0, 3, 6, 9]] ** 2, dim=1)
        return action_l2, action_l2

    def _reward_leg_energy_abs_sum(self):
        energy = torch.sum(torch.abs(self.env.torques[:, :12] * self.env.dof_vel[:, :12]), dim = 1)
        return energy, energy

    def _reward_leg_energy_sum_abs(self):
        energy = torch.abs(torch.sum(self.env.torques[:, :12] * self.env.dof_vel[:, :12], dim = 1))
        return energy, energy
    
    def _reward_leg_action_l2(self):
        action_l2 = torch.sum(self.env.actions[:, :12] ** 2, dim=1)
        return action_l2, action_l2
    
    def _reward_leg_energy(self):
        energy = torch.sum(self.env.torques[:, :12] * self.env.dof_vel[:, :12], dim = 1)
        return energy, energy
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.env.cfg.rewards.tracking_sigma), lin_vel_error

    def _reward_tracking_lin_vel_x_l1(self):
        zero_cmd_indices = torch.abs(self.env.commands[:, 0]) < 1e-5
        error = torch.abs(self.env.commands[:, 0] - self.env.base_lin_vel[:, 0])
        rew = 0*error
        rew_x = -error + torch.abs(self.env.commands[:, 0])
        rew[~zero_cmd_indices] = rew_x[~zero_cmd_indices] / (torch.abs(self.env.commands[~zero_cmd_indices, 0]) + 0.01)
        rew[zero_cmd_indices] = 0
        return rew, error

    def _reward_tracking_lin_vel_x_exp(self):
        error = torch.abs(self.env.commands[:, 0] - self.env.base_lin_vel[:, 0])
        return torch.exp(-error/self.env.cfg.rewards.tracking_sigma), error

    def _reward_tracking_ang_vel_yaw_l1(self):
        error = torch.abs(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return - error + torch.abs(self.env.commands[:, 2]), error
    
    def _reward_tracking_ang_vel_yaw_exp(self):
        error = torch.abs(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-error/self.env.cfg.rewards.tracking_sigma), error

    def _reward_tracking_lin_vel_y_l2(self):
        squared_error = (self.env.commands[:, 1] - self.env.base_lin_vel[:, 1]) ** 2
        return squared_error, squared_error
    
    def _reward_tracking_lin_vel_z_l2(self):
        squared_error = (self.env.commands[:, 2] - self.env.base_lin_vel[:, 2]) ** 2
        return squared_error, squared_error
    
    def _reward_survive(self):
        survival_reward = torch.ones(self.env.num_envs, device=self.env.device)
        return survival_reward, survival_reward

    def _reward_foot_contacts_z(self):
        foot_contacts_z = torch.square(self.env.force_sensor_tensor[:, :, 2]).sum(dim=-1)
        return foot_contacts_z, foot_contacts_z

    def _reward_torques(self):
        # Penalize torques
        torque = torch.sum(torch.square(self.env.torques), dim=1)
        return torque, torque
    
    def _reward_energy_square(self):
        energy = torch.sum(torch.square(self.env.torques[:, :12] * self.env.dof_vel[:, :12]), dim=1)
        return energy, energy

    def _reward_tracking_lin_vel_y(self):
        cmd = self.env.commands[:, 1].clone()
        lin_vel_y_error = torch.square(cmd - self.env.base_lin_vel[:, 1])
        rew = torch.exp(-lin_vel_y_error/self.env.cfg.rewards.tracking_sigma)
        return rew, lin_vel_y_error
    
    def _reward_lin_vel_z(self):
        rew = torch.square(self.env.base_lin_vel[:, 2])
        return rew, rew
    
    def _reward_ang_vel_xy(self):
        rew = torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)
        return rew, rew
    
    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.env.cfg.rewards.tracking_sigma), ang_vel_error
    
    def _reward_work(self):
        work = self.env.torques * self.env.dof_vel
        abs_sum_work = torch.abs(torch.sum(work[:, :12], dim = 1))
        return abs_sum_work, abs_sum_work
    
    def _reward_dof_acc(self):
        rew = torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel)[:, :12] / self.env.dt), dim=1)
        return rew, rew
    
    def _reward_action_rate(self):
        action_rate = torch.sum(torch.square(self.env.last_actions - self.env.actions)[:, :12], dim=1)
        return action_rate, action_rate
    
    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.) # upper limit
        rew = torch.sum(out_of_limits[:, :12], dim=1)
        return rew, rew
    
    def _reward_delta_torques(self):
        rew = torch.sum(torch.square(self.env.torques - self.env.last_torques)[:, :12], dim=1)
        return rew, rew
    
    def _reward_collision(self):
        rew = torch.sum(1.*(torch.norm(self.env.contact_forces[:, self.env.penalized_contact_indices, :], dim=-1) > 0.1), dim=1)
        return rew, rew
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        dof_error = torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        rew = torch.exp(-dof_error*0.05)
        rew[self.env._get_walking_cmd_mask()] = 0.
        return rew, rew

    def _reward_walking_dof(self):
        # Penalize motion at zero commands
        dof_error = torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        rew = torch.exp(-dof_error*0.05)
        rew[~self.env._get_walking_cmd_mask()] = 0.
        return rew, rew

    def _reward_hip_pos(self):
        rew = torch.sum(torch.square(self.env.dof_pos[:, self.env.hip_indices] - self.env.default_dof_pos[self.env.hip_indices]), dim=1)
        return rew, rew

    def _reward_feet_jerk(self):
        if not hasattr(self, "last_contact_forces"):            
            result = torch.zeros(self.env.num_envs).to(self.env.device)
        else:
            result = torch.sum(torch.norm(self.env.force_sensor_tensor - self.env.last_contact_forces, dim=-1), dim=-1)
        
        self.env.last_contact_forces = self.env.force_sensor_tensor.clone()
        result[self.env.episode_length_buf<50] = 0.
        return result, result
    
    # ============ Stair Climbing Specialized Reward Functions ============
    
    def _reward_stair_climbing_progressive(self):
        """Progressive stair climbing reward - Core reward with exponential growth mechanism"""
        current_height = self.env.root_states[:, 2]
        
        if not hasattr(self.env, 'initial_base_height'):
            self.env.initial_base_height = current_height.clone()
            self.env.max_achieved_height = current_height.clone()
        
        height_gain = current_height - self.env.initial_base_height
        
        self.env.max_achieved_height = torch.maximum(self.env.max_achieved_height, current_height)
        total_progress = self.env.max_achieved_height - self.env.initial_base_height
        
        base_reward = torch.clamp(height_gain * 5.0, min=0.0, max=10.0)
        exponential_reward = torch.clamp(torch.exp(height_gain * 2.0) - 1.0, min=0.0, max=50.0)
        persistence_reward = torch.clamp(total_progress * 3.0, min=0.0, max=20.0)
        
        forward_command_mask = self.env.commands[:, 0] > 0.1
        total_reward = (base_reward + exponential_reward + persistence_reward) * forward_command_mask.float()
        
        return total_reward, height_gain
    
    def _reward_stair_velocity_tracking_enhanced(self):
        """Enhanced velocity tracking reward - GO2 style design optimized for stairs"""
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        
        tracking_reward = torch.exp(-lin_vel_error / (self.env.cfg.rewards.tracking_sigma * 0.8))
        
        forward_stability = torch.exp(-torch.abs(self.env.base_lin_vel[:, 0] - self.env.commands[:, 0]) / 0.2)
        
        moving_mask = torch.norm(self.env.commands[:, :2], dim=1) > 0.05
        enhanced_reward = (tracking_reward * 0.7 + forward_stability * 0.3) * moving_mask.float()
        
        return enhanced_reward, lin_vel_error
    
    def _reward_stair_base_height_adaptive(self):
        """Adaptive base height control - GO2 style adapted for stairs"""
        current_height = self.env.root_states[:, 2]
        
        if hasattr(self.env, 'measured_heights') and torch.is_tensor(self.env.measured_heights):
            terrain_height = torch.mean(self.env.measured_heights * self.env.cfg.terrain.vertical_scale, dim=1)
            adaptive_target = terrain_height + self.env.cfg.rewards.base_height_target
        else:
            adaptive_target = torch.full_like(current_height, self.env.cfg.rewards.base_height_target)
        
        height_error = torch.square(current_height - adaptive_target)
        
        upward_tolerance = current_height > adaptive_target
        modified_error = torch.where(upward_tolerance, height_error * 0.3, height_error)
        
        return modified_error, height_error
    
    def _reward_stair_feet_air_time_enhanced(self):
        """Enhanced feet air time reward - GO2 style adapted for stair gait"""
        contact_thresh = 5.0
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > contact_thresh
        
        if not hasattr(self.env, 'feet_air_time_stairs'):
            self.env.feet_air_time_stairs = torch.zeros_like(contact, dtype=torch.float)
            self.env.last_contacts_stairs = torch.zeros_like(contact, dtype=torch.bool)
        
        self.env.feet_air_time_stairs += self.env.dt
        
        contact_filt = torch.logical_or(contact, self.env.last_contacts_stairs)
        first_contact = (self.env.feet_air_time_stairs > 0.) * contact_filt
        
        optimal_air_time = 0.5
        air_time_quality = torch.exp(-torch.abs(self.env.feet_air_time_stairs - optimal_air_time) / 0.2)
        rew_airTime = torch.sum(air_time_quality * first_contact, dim=1)
        
        moving_mask = torch.norm(self.env.commands[:, :2], dim=1) > 0.1
        rew_airTime *= moving_mask.float()
        
        self.env.feet_air_time_stairs *= ~contact_filt
        self.env.last_contacts_stairs = contact
        
        return rew_airTime, torch.mean(self.env.feet_air_time_stairs, dim=1)
    
    def _reward_stair_foot_clearance_enhanced(self):
        """Enhanced foot clearance reward - Prevent collision with stair edges"""
        if not hasattr(self.env, 'measured_heights') or not torch.is_tensor(self.env.measured_heights):
            return torch.zeros(self.env.num_envs, device=self.env.device), torch.zeros(self.env.num_envs, device=self.env.device)
        
        feet_pos = self.env.rigid_body_state[:, self.env.feet_indices, :3]
        base_pos = self.env.root_states[:, :3].unsqueeze(1)
        
        relative_feet_pos = feet_pos - base_pos
        feet_height_above_base = relative_feet_pos[:, :, 2]
        
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 5.0
        in_air = ~contact
        
        target_clearance = 0.08
        clearance_error = torch.abs(feet_height_above_base - target_clearance)
        
        air_clearance_reward = torch.exp(-clearance_error * 10.0) * in_air.float()
        total_clearance_reward = torch.sum(air_clearance_reward, dim=1)
        
        moving_mask = torch.norm(self.env.commands[:, :2], dim=1) > 0.05
        total_clearance_reward *= moving_mask.float()
        
        return total_clearance_reward, torch.mean(clearance_error, dim=1)
    
    def _reward_stair_stability_enhanced(self):
        """Enhanced stair stability reward - Balance climbing needs with stability"""
        base_euler = torch.stack(euler_from_quat(self.env.base_quat), dim=-1)
        roll, pitch, yaw = base_euler[:, 0], base_euler[:, 1], base_euler[:, 2]
        
        roll_penalty = torch.square(roll)
        
        optimal_pitch = 0.1
        pitch_error = torch.square(pitch - optimal_pitch)
        
        stability_reward = torch.exp(-(roll_penalty * 8.0 + pitch_error * 4.0))
        orientation_error = roll_penalty + pitch_error
        
        return stability_reward, orientation_error
    
    def _reward_stair_forward_progress(self):
        """Stair forward progress reward - Encourage continuous forward climbing"""
        if not hasattr(self.env, 'cumulative_forward_distance'):
            self.env.cumulative_forward_distance = torch.zeros(self.env.num_envs, device=self.env.device)
            self.env.last_forward_pos = self.env.root_states[:, 0].clone()
        
        current_forward_pos = self.env.root_states[:, 0]
        forward_delta = current_forward_pos - self.env.last_forward_pos
        
        positive_forward = torch.clamp(forward_delta, min=0.0)
        self.env.cumulative_forward_distance += positive_forward
        
        linear_reward = positive_forward * 10.0
        cumulative_bonus = torch.clamp(self.env.cumulative_forward_distance * 2.0, max=30.0)
        
        total_forward_reward = linear_reward + cumulative_bonus * 0.1
        
        self.env.last_forward_pos = current_forward_pos.clone()
        
        return total_forward_reward, self.env.cumulative_forward_distance
    
    def _reward_stair_energy_efficiency_v2(self):
        """Stair energy efficiency reward V2 - Balance performance with efficiency"""
        leg_torques = self.env.torques[:, :12]
        torque_penalty = torch.sum(torch.square(leg_torques), dim=1)
        
        if hasattr(self.env, 'last_actions'):
            action_smoothness = torch.sum(torch.square(self.env.actions[:, :12] - self.env.last_actions[:, :12]), dim=1)
        else:
            action_smoothness = torch.zeros(self.env.num_envs, device=self.env.device)
        
        efficiency_reward = torch.exp(-(torque_penalty * 1e-6 + action_smoothness * 0.01))
        total_penalty = torque_penalty * 0.3 + action_smoothness * 0.7
        
        return efficiency_reward, total_penalty
    
    def _reward_alive(self):
        return 1., 1.
    
    def _reward_feet_drag(self):
        feet_xyz_vel = torch.abs(self.env.rigid_body_state[:, self.env.feet_indices, 7:10]).sum(dim=-1)
        dragging_vel = self.env.foot_contacts_from_sensor * feet_xyz_vel
        rew = dragging_vel.sum(dim=-1)
        return rew, rew

    def _reward_feet_contact_forces(self):
        reset_flag = (self.env.episode_length_buf > 2./self.env.dt).type(torch.float)
        forces = torch.sum((torch.norm(self.env.force_sensor_tensor, dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0), dim=-1)
        rew = reset_flag * forces
        return rew, rew
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        error = torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)
        return error, error
    
    def _reward_roll(self):
        # Penalize non flat base orientation
        roll = self.env._get_body_orientation()[:, 0]
        error = torch.abs(roll)
        return error, error
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.env.root_states[:, 2].unsqueeze(1), dim=1)
        return torch.abs(base_height - self.env.cfg.rewards.base_height_target), base_height
    
    def _reward_orientation_walking(self):
        reward = self.env._reward_orientation()
        metric = torch.zeros_like(reward)
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_orientation_standing(self):
        reward = self.env._reward_orientation()
        metric = torch.zeros_like(reward)
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_torques_walking(self):
        reward = self.env._reward_torques()
        metric = torch.zeros_like(reward)
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_torques_standing(self):
        reward = self.env._reward_torques()
        metric = torch.zeros_like(reward)
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_energy_square_walking(self):
        reward, metric = self._reward_energy_square()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_energy_square_standing(self):
        reward, metric = self._reward_energy_square()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_base_height_walking(self):
        reward = self.env._reward_base_height()
        metric = torch.zeros_like(reward)
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_base_height_standing(self):
        reward = self.env._reward_base_height()
        metric = torch.zeros_like(reward)
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric
    
    def _reward_dof_default_pos(self):
        dof_error = torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        rew = torch.exp(-dof_error*0.05)
        
        return rew, rew
    
    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        return dof_error, dof_error
    
    def _reward_tracking_lin_vel_max(self):
        rew = torch.where(self.env.commands[:, 0] > 0, torch.minimum(self.env.base_lin_vel[:, 0], self.env.commands[:, 0]) / (self.env.commands[:, 0] + 1e-5), \
                          torch.minimum(-self.env.base_lin_vel[:, 0], -self.env.commands[:, 0]) / (-self.env.commands[:, 0] + 1e-5))
        zero_cmd_indices = torch.abs(self.env.commands[:, 0]) < self.env.cfg.commands.lin_vel_x_clip
        rew[zero_cmd_indices] = torch.exp(-torch.abs(self.env.base_lin_vel[:, 0]))[zero_cmd_indices]
        return rew, rew
    
    def _reward_penalty_lin_vel_y(self):
        rew = torch.abs(self.env.base_lin_vel[:, 1])
        rot_indices = torch.abs(self.env.commands[:, 2]) > self.env.cfg.commands.ang_vel_yaw_clip
        rew[rot_indices] = 0.
        return rew, rew
    
    # -------------B1 Gait Control Rewards----------------
    def _reward_tracking_contacts_shaped_force(self):
        if not self.env.cfg.env.observe_gait_commands:
            return 0,0
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
        
        # cmd_stop_flag = ~self.env._get_walking_cmd_mask()
        # reward[cmd_stop_flag] = 0
        return reward / 4, reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        if not self.env.cfg.env.observe_gait_commands:
            return 0,0
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
        # cmd_stop_flag = ~self.env._get_walking_cmd_mask()
        # reward[cmd_stop_flag] = 0
        
        return reward / 4, reward / 4
    
    def _reward_feet_height(self):
        feet_height_tracking = self.env.cfg.rewards.feet_height_target

        if self.env.cfg.rewards.feet_height_allfeet:
            feet_height = self.env.rigid_body_state[:, self.env.feet_indices, 2] # All feet
        else:
            feet_height = self.env.rigid_body_state[:, self.env.feet_indices[:2], 2] # Only front feet

        rew = torch.clamp(torch.norm(feet_height, dim=-1) - feet_height_tracking, max=0)
        cmd_stop_flag = ~self.env._get_walking_cmd_mask()
        rew[cmd_stop_flag] = 0
        return rew, rew

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        first_contact = (self.env.feet_air_time > 0.) * self.env.foot_contacts_from_sensor  #self.env.contact_filt
        self.env.feet_air_time += self.env.dt

        if self.env.cfg.rewards.feet_aritime_allfeet:
            rew_airTime = torch.sum((self.env.feet_air_time - 0.5) * first_contact, dim=1)
        else:
            rew_airTime = torch.sum((self.env.feet_air_time[:, :2] - 0.5) * first_contact[:, :2], dim=1)
        
        rew_airTime *= self.env._get_walking_cmd_mask()  # reward for stepping for any of the 3 motions
        self.env.feet_air_time *= ~ self.env.foot_contacts_from_sensor  #self.env.contact_filt
        return rew_airTime, rew_airTime