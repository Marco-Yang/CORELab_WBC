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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class B1Z1StairsRandomCfg( LeggedRobotCfg ):
    class goal_ee:
        num_commands = 3
        traj_time = [1, 3]
        hold_time = [0.5, 2]
        collision_upper_limits = [0.1, 0.2, -0.05]
        collision_lower_limits = [-0.8, -0.2, -0.7]
        underground_limit = -0.7
        num_collision_check_samples = 10
        command_mode = 'sphere'
        arm_induced_pitch = 0.38 # Added to -pos_p (negative goal pitch) to get default eef orn_p

        class sphere_center:
            x_offset = 0.3 # Relative to base
            y_offset = 0 # Relative to base
            z_invariant_offset = 0.7 # Relative to terrain

        class ranges:
            init_pos_start = [0.5, np.pi/8, 0]
            init_pos_end = [0.7, 0, 0]
            pos_l = [0.4, 0.95]
            pos_p = [-1 * np.pi / 2.5, 1 * np.pi / 3]
            pos_y = [-1.2, 1.2]
            
            delta_orn_r = [-0.5, 0.5]
            delta_orn_p = [-0.5, 0.5]
            delta_orn_y = [-0.5, 0.5]
            final_tracking_ee_reward = 0.55

        sphere_error_scale = [1, 1, 1]#[1 / (ranges.final_pos_l[1] - ranges.final_pos_l[0]), 1 / (ranges.final_pos_p[1] - ranges.final_pos_p[0]), 1 / (ranges.final_pos_y[1] - ranges.final_pos_y[0])]
        orn_error_scale = [1, 1, 1]#[2 / np.pi, 2 / np.pi, 2 / np.pi]

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1   

    class commands:
        curriculum = True
        num_commands = 3
        resampling_time = 3.

        lin_vel_x_schedule = [0, 1.5]
        ang_vel_yaw_schedule = [0, 1.1]
        tracking_ang_vel_yaw_schedule = [0, 1.1]

        ang_vel_yaw_clip = 0.9
        lin_vel_x_clip = 1.0

        class ranges:
            lin_vel_x = [-0.6, 1.5]
            ang_vel_yaw = [-1.1, 1.1]

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel =  1.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class env:
        num_envs = 1
        num_actions = 12 + 6 #CAUTION
        num_torques = 12 + 6
        action_delay = 2
        num_gripper_joints = 1
        num_proprio = 66
        num_priv = 18
        history_len = 10
        num_observations = num_proprio * (history_len+1) + num_priv
        num_privileged_obs = None
        send_timeouts = True
        episode_length_s = 66
        reorder_dofs = True
        teleop_mode = False
        record_video = False
        stand_by = False
        observe_gait_commands = True
        frequencies = 2
        stop_update_goal = False

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.2,   # [rad]
            'FL_thigh_joint': 0.8,     # [rad]
            'FL_calf_joint': -1.5,   # [rad]

            'RL_hip_joint': 0.2,   # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]

            'FR_hip_joint': -0.2 ,  # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'FR_calf_joint': -1.5,  # [rad]

            'RR_hip_joint': -0.2,   # [rad]
            'RR_thigh_joint': 0.8,   # [rad]
            'RR_calf_joint': -1.5,    # [rad]

            'z1_waist': 0.0,
            'z1_shoulder': 1.48,
            'z1_elbow': -0.63,
            'z1_wrist_angle': -0.84,
            'z1_forearm_roll': 0.0,
            'z1_wrist_rotate': 1.57,
            'z1_jointGripper': -0.785,
        }
        rand_yaw_range = np.pi/2
        origin_perturb_range = 0.5
        init_vel_perturb_range = 0.1

    class control:
        stiffness = {'joint': 85, 'z1': 5}
        damping = {'joint': 2.5, 'z1': 0.5}

        adaptive_arm_gains = False
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = [0.35, 0.40, 0.40] * 2 + [0.35, 0.40, 0.40] * 2 + [2.1, 0.6, 0.6, 0, 0, 0]  
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        torque_supervision = False

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/b1z1/urdf/b1z1.urdf'
        foot_name = "foot"
        gripper_name = "ee_gripper_link" #"gripperMover"
        penalize_contacts_on = ["thigh", "trunk", "calf"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        collapse_fixed_joints = True # Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False
    
    class box:
        box_size = 0.1
        randomize_base_mass = True
        added_mass_range = [-0.001, 0.050]
        box_env_origins_x = 0
        box_env_origins_y_range = [0.1, 0.3]
        box_env_origins_z = box_size / 2 + 0.16
    
    class arm:
        init_target_ee_base = [0.2, 0.0, 0.2]
        grasp_offset = 0.08
        osc_kp = np.array([100, 100, 100, 30, 30, 30])
        osc_kd = 2 * (osc_kp ** 0.5)

    class domain_rand:
        observe_priv = True
        randomize_friction = True
        friction_range = [0.3, 3.0] # [0.5, 3.0]
        randomize_base_mass = True
        added_mass_range = [0., 15.]
        randomize_base_com = True
        added_com_range_x = [-0.15, 0.15]
        added_com_range_y = [-0.15, 0.15]
        added_com_range_z = [-0.15, 0.15]
        randomize_motor = True
        leg_motor_strength_range = [0.7, 1.3]
        arm_motor_strength_range = [0.7, 1.3]
        randomize_gripper_mass = True
        gripper_added_mass_range = [0, 0.1]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5
  
    class rewards:
        reward_container_name = "maniploco_rewards"

        only_positive_rewards = False
        tracking_sigma = 0.25
        tracking_ee_sigma = 1.5
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.55
        base_height_target = 0.62
        max_contact_force = 50.
        gait_vel_sigma = 0.35
        gait_force_sigma = 0.35
        kappa_gait_probs = 0.08
        feet_height_target = 0.38

        feet_aritime_allfeet = True
        feet_height_allfeet = True

        class scales:
            tracking_lin_vel_max = 1.2
            feet_air_time = 2.5
            base_height = -2.0
            orientation = -0.6
            
            tracking_contacts_shaped_force = -1.2
            tracking_contacts_shaped_vel = -1.2
            feet_height = 2.2

            tracking_lin_vel_x_l1 = 0.0
            tracking_lin_vel_x_exp = 0.0
            tracking_ang_vel = 0.9

            delta_torques = -3.5e-8
            work = -0.0032
            energy_square = -7.0e-7
            torques = -2.8e-5
            
            stand_still = 0.5
            walking_dof = 2.2
            dof_default_pos = -0.18
            dof_error = -0.10
            alive = 2.0
            lin_vel_z = -2.2
            roll = -2.8

            ang_vel_xy = -0.35
            dof_acc = -8.0e-7
            collision = -20.0
            action_rate = -0.020
            dof_pos_limits = -12.0
            hip_pos = -0.4
            feet_jerk = -0.00035
            feet_drag = -0.10
            feet_contact_forces = -0.0015
            
            orientation_walking = 0.4
            orientation_standing = 0.0
            base_height_walking = 0.7
            base_height_standing = 0.0
            torques_walking = -2.8e-5
            torques_standing = 0.0
            energy_square_walking = -7.0e-7
            energy_square_standing = 0.0
            penalty_lin_vel_y = -0.5

        class arm_scales:
            arm_termination = None
            tracking_ee_sphere = 0.0
            tracking_ee_world = 0.10
            tracking_ee_sphere_walking = 0.0
            tracking_ee_sphere_standing = 0.0
            tracking_ee_cart = None
            arm_orientation = 0.0
            arm_energy_abs_sum = -0.002
            tracking_ee_orn = 0.0
            tracking_ee_orn_ry = None
        
    class viewer:
        pos = [-20, 0, 20]  # [m]
        lookat = [0, 0, -2]  # [m]

    class termination:
        r_threshold = 0.8
        p_threshold = 0.8
        z_threshold = 0.1

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "fast"  # grid or fast
        max_error = 0.1 # for fast
        horizontal_scale = 0.05 # [m] influence computation time by a lot
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        height = [0.05, 0.18]
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = False

        all_vertical = False
        no_flat = False
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 9
        terrain_length = 24.
        terrain_width = 24.
        num_rows= 10
        num_cols = 5

        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.,
                        "rough slope down": 0.,
                        "rough stairs up": 0.0,
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.,
                        "gaps": 0., 
                        "rough flat": 0.0,
                        "pit": 0.0,
                        "wall": 0.0,
                        "random stairs up": 1.0}
        terrain_proportions = list(terrain_dict.values())
        slope_treshold = 0.75
        origin_zero_z = False


class B1Z1StairsRandomCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        continue_from_last_std = True
        init_std = [[0.8, 1.0, 1.0] * 4 + [1.0] * 6]
        actor_hidden_dims = [128]
        critic_hidden_dims = [128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        output_tanh = False

        leg_control_head_hidden_dims = [128, 128]
        arm_control_head_hidden_dims = [128, 128]

        priv_encoder_dims = [64, 20]

        num_leg_actions = 12
        num_arm_actions = 6

        adaptive_arm_gains = B1Z1StairsRandomCfg.control.adaptive_arm_gains
        adaptive_arm_gains_scale = 10.0
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1e-4
        schedule = 'fixed'
        gamma = 0.99
        lam = 0.95
        desired_kl = None
        max_grad_norm = 1.
        min_policy_std = [[0.15, 0.25, 0.25] * 4 + [0.2] * 3 + [0.05] * 3]

        mixing_schedule=[1.0, 0, 3000] #if not RESUME else [1.0, 0, 1]
        torque_supervision = B1Z1StairsRandomCfg.control.torque_supervision  #alert: also appears above
        torque_supervision_schedule=[0.0, 1000, 1000]
        adaptive_arm_gains = B1Z1StairsRandomCfg.control.adaptive_arm_gains
        # dagger params
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 3000, 7000] #if not RESUME else [0, 1, 1000, 1000]

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 80000 # number of policy updates
        # logging
        save_interval = 200 # check for potential saves every this many iterations
        experiment_name = 'b1z1_v2'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

