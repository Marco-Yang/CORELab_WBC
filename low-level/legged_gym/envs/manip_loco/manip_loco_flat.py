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

from .manip_loco import ManipLoco as OriginalManipLoco
from .b1z1_flat_config import B1Z1FlatCfg
from legged_gym.utils.terrain import Terrain
import torch
import numpy as np

class ManipLocoFlat(OriginalManipLoco):
    """
    Flat ground version of ManipLoco environment.
    Inherits all functionality from ManipLoco but uses flat terrain configuration.
    """
    cfg : B1Z1FlatCfg
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
    
    def create_sim(self):
        """ Creates simulation, terrain and environments - override to use flat ground
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.terrain = Terrain(self.cfg.terrain)
        self._create_ground_plane()
        self._create_envs()
    
    def _get_env_origins(self):
        """ Set environment origins. For flat ground, all origins are at [0, 0, 0]
        """
        import numpy as np
        env_origins_np = np.zeros((self.cfg.env.num_envs, 3))
        spacing = 4.
        rows = int(np.sqrt(self.cfg.env.num_envs))
        cols = int(np.ceil(self.cfg.env.num_envs / rows))
        
        for i in range(self.cfg.env.num_envs):
            row = i // cols
            col = i % cols
            env_origins_np[i, 0] = row * spacing
            env_origins_np[i, 1] = col * spacing
            env_origins_np[i, 2] = 0.0
        
        self.env_origins = torch.from_numpy(env_origins_np).to(self.device).to(torch.float)
        self.terrain_origins = self.env_origins.clone()