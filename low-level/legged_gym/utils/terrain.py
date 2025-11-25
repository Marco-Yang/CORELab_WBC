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
from numpy.random import choice
from scipy import interpolate
import random

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

import matplotlib.pyplot as plt
from pydelatin import Delatin

class Terrain_Perlin:
    def __init__(self, cfg):
        self.cfg = cfg
        self.xSize = int(cfg.horizontal_scale * cfg.tot_cols)
        self.ySize = int(cfg.horizontal_scale * cfg.tot_rows)
        assert(self.xSize == cfg.horizontal_scale * cfg.tot_cols and self.ySize == cfg.horizontal_scale * cfg.tot_rows)
        self.tot_cols = cfg.tot_cols
        self.tot_rows = cfg.tot_rows
        self.heightsamples_float = self.generate_fractal_noise_2d(self.xSize, self.ySize, self.tot_cols, self.tot_rows, zScale=cfg.zScale)
        # self.heightsamples_float[self.tot_cols//2 - 40: self.tot_cols//2 + 40, :] = np.mean(self.heightsamples_float)
        self.heightsamples = (self.heightsamples_float * (1 / cfg.vertical_scale)).astype(np.int16)
        print("Terrain heightsamples shape: ", self.heightsamples.shape)
        print("Terrain heightsamples stat: ", np.array([np.min(self.heightsamples), np.max(self.heightsamples), np.mean(self.heightsamples), np.std(self.heightsamples), np.median(self.heightsamples)]) * cfg.vertical_scale)
        # self.heightsamples = np.zeros((800, 800)).astype(np.int16)
        self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.heightsamples,
                                                                                        cfg.horizontal_scale,
                                                                                        cfg.vertical_scale,
                                                                                        cfg.slope_treshold)
    
    def generate_perlin_noise_2d(self, shape, res):
        def f(t):
            return 6*t**5 - 15*t**4 + 10*t**3

        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
        # Gradients
        angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
        g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
        # Ramps
        n00 = np.sum(grid * g00, 2)
        n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
        # Interpolation
        t = f(grid)
        n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
        n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
        return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1) * 0.5 + 0.5
    
    def generate_fractal_noise_2d(self, xSize=20, ySize=20, xSamples=1600, ySamples=1600, \
        frequency=10, fractalOctaves=2, fractalLacunarity = 2.0, fractalGain=0.25, zScale = 0.23):
        xScale = frequency * xSize
        yScale = frequency * ySize
        amplitude = 1
        shape = (xSamples, ySamples)
        noise = np.zeros(shape)
        for _ in range(fractalOctaves):
            noise += amplitude * self.generate_perlin_noise_2d((xSamples, ySamples), (xScale, yScale)) * zScale
            amplitude *= fractalGain
            xScale, yScale = int(fractalLacunarity * xScale), int(fractalLacunarity * yScale)

        return noise

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain) -> None:

        self.cfg = cfg
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        cfg.terrain_proportions = np.array(cfg.terrain_proportions) / np.sum(cfg.terrain_proportions)
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        # self.env_slope_vec = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.curiculum(random=True)
            # self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            print("Converting heightmap to trimesh...")
            if cfg.hf2mesh_method == "grid":
                self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                                self.cfg.horizontal_scale,
                                                                                                self.cfg.vertical_scale,
                                                                                                self.cfg.slope_treshold)
            else:
                assert cfg.hf2mesh_method == "fast", "Height field to mesh method must be grid or fast"
                self.vertices, self.triangles = convert_heightfield_to_trimesh_delatin(self.height_field_raw, self.cfg.horizontal_scale, self.cfg.vertical_scale, max_error=cfg.max_error)
            print("Created {} vertices".format(self.vertices.shape[0]))
            print("Created {} triangles".format(self.triangles.shape[0]))

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            # difficulty = np.random.choice([0.5, 0.75, 0.9])
            difficulty = np.random.uniform(-0.2, 1.2)
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self, random=False):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                if self.cfg.num_rows == 1:
                    difficulty = 0.5
                else:
                    difficulty = i / (self.cfg.num_rows-1)
                choice = j / self.cfg.num_cols + 0.001
                if random:
                    terrain = self.make_terrain(choice, np.random.uniform(0, 1))
                else:
                    terrain = self.make_terrain(choice, difficulty)

                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        print(f"Generating selected terrain type: {terrain_type}")
        
        for k in range(self.cfg.num_sub_terrains):
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.cfg.vertical_scale,
                              horizontal_scale=self.cfg.horizontal_scale)

            if terrain_type == 'pyramid_stairs_terrain_wrapper':
                step_width = self.cfg.terrain_kwargs['terrain_kwargs']['step_width']
                step_height = self.cfg.terrain_kwargs['terrain_kwargs']['step_height']
                platform_size = self.cfg.terrain_kwargs['terrain_kwargs']['platform_size']
                print(f"   Generating stairs [{i},{j}]: width={step_width}, height={step_height}, platform={platform_size}")
                terrain_utils.pyramid_stairs_terrain(terrain, step_width=step_width, step_height=step_height, platform_size=platform_size)
            else:
                eval(terrain_type)(terrain, **self.cfg.terrain_kwargs['terrain_kwargs'])
            
            self.add_terrain_to_map(terrain, i, j)
    
    def add_roughness(self, terrain, difficulty=1):
        max_height = (self.cfg.height[1] - self.cfg.height[0]) * difficulty + self.cfg.height[0]
        height = random.uniform(self.cfg.height[0], max_height)
        terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale)
    
    def _create_random_height_stairs(self, terrain, step_width, min_height, max_height, platform_size=1.):
        """
        Generate four-sided upward random height inverted pyramid stairs environment.
        Robot starts from center platform, can climb stairs in all four directions.
        Each step height is random within [min_height, max_height] range.
        """
        step_width_px = int(step_width / terrain.horizontal_scale)
        platform_size_px = int(platform_size / terrain.horizontal_scale)
        
        print(f"Creating four-sided random height stairs: step_width={step_width}m ({step_width_px}px), height_range=[{min_height}, {max_height}]m, platform_size={platform_size}m ({platform_size_px}px)")
        
        center_x = terrain.width // 2
        center_y = terrain.length // 2
        
        max_possible_height = int(max_height * 20 / terrain.vertical_scale)
        terrain.height_field_raw[:, :] = max_possible_height
        
        platform_start_x = center_x - platform_size_px // 2
        platform_end_x = center_x + platform_size_px // 2
        platform_start_y = center_y - platform_size_px // 2
        platform_end_y = center_y + platform_size_px // 2
        
        platform_start_x = max(0, platform_start_x)
        platform_end_x = min(terrain.width, platform_end_x)
        platform_start_y = max(0, platform_start_y)
        platform_end_y = min(terrain.length, platform_end_y)
        
        max_layers_x = min((platform_start_x) // step_width_px, (terrain.width - platform_end_x) // step_width_px)
        max_layers_y = min((platform_start_y) // step_width_px, (terrain.length - platform_end_y) // step_width_px)
        max_layers = min(max_layers_x, max_layers_y)
        
        print(f"Max layers: {max_layers}, platform range: ({platform_start_x}-{platform_end_x}, {platform_start_y}-{platform_end_y})")
        
        terrain.height_field_raw[platform_start_x:platform_end_x, platform_start_y:platform_end_y] = 0
        
        cumulative_height = 0
        
        for layer in range(1, max_layers + 1):
            random_step_height = np.random.uniform(min_height, max_height)
            step_height_px = int(random_step_height / terrain.vertical_scale)
            cumulative_height += step_height_px
            
            outer_start_x = platform_start_x - layer * step_width_px
            outer_end_x = platform_end_x + layer * step_width_px
            outer_start_y = platform_start_y - layer * step_width_px
            outer_end_y = platform_end_y + layer * step_width_px
            
            inner_start_x = platform_start_x - (layer - 1) * step_width_px
            inner_end_x = platform_end_x + (layer - 1) * step_width_px
            inner_start_y = platform_start_y - (layer - 1) * step_width_px
            inner_end_y = platform_end_y + (layer - 1) * step_width_px
            
            outer_start_x = max(0, outer_start_x)
            outer_end_x = min(terrain.width, outer_end_x)
            outer_start_y = max(0, outer_start_y)
            outer_end_y = min(terrain.length, outer_end_y)
            
            inner_start_x = max(0, inner_start_x)
            inner_end_x = min(terrain.width, inner_end_x)
            inner_start_y = max(0, inner_start_y)
            inner_end_y = min(terrain.length, inner_end_y)
            
            terrain.height_field_raw[outer_start_x:outer_end_x, outer_start_y:inner_start_y] = cumulative_height
            terrain.height_field_raw[outer_start_x:outer_end_x, inner_end_y:outer_end_y] = cumulative_height
            terrain.height_field_raw[outer_start_x:inner_start_x, inner_start_y:inner_end_y] = cumulative_height
            terrain.height_field_raw[inner_end_x:outer_end_x, inner_start_y:inner_end_y] = cumulative_height
            
            print(f"Layer {layer}: outer({outer_start_x}-{outer_end_x}, {outer_start_y}-{outer_end_y}), inner({inner_start_x}-{inner_end_x}, {inner_start_y}-{inner_end_y}), height_increment={random_step_height:.3f}m, cumulative_height={cumulative_height * terrain.vertical_scale:.3f}m")
        
        return terrain

    def _create_inverted_pyramid_stairs(self, terrain, step_width, step_height, platform_size=1.):
        """
        Generate upward stairs environment with center depression and raised edges (inverted pyramid stairs).
        Each step has uniform width and height, center depression aligns with edges.
        """
        step_width_px = int(step_width / terrain.horizontal_scale)
        step_height_px = int(step_height / terrain.vertical_scale)
        platform_size_px = int(platform_size / terrain.horizontal_scale)
        
        print(f"Creating inverted pyramid stairs: step_width={step_width}m ({step_width_px}px), step_height={step_height}m ({step_height_px}px), platform_size={platform_size}m ({platform_size_px}px)")
        
        center_x = terrain.width // 2
        center_y = terrain.length // 2
        
        terrain.height_field_raw[:, :] = 0
        
        platform_start_x = center_x - platform_size_px // 2
        platform_end_x = center_x + platform_size_px // 2
        platform_start_y = center_y - platform_size_px // 2
        platform_end_y = center_y + platform_size_px // 2
        
        platform_start_x = max(0, platform_start_x)
        platform_end_x = min(terrain.width, platform_end_x)
        platform_start_y = max(0, platform_start_y)
        platform_end_y = min(terrain.length, platform_end_y)
        
        max_layers_x = min((platform_start_x) // step_width_px, (terrain.width - platform_end_x) // step_width_px)
        max_layers_y = min((platform_start_y) // step_width_px, (terrain.length - platform_end_y) // step_width_px)
        max_layers = min(max_layers_x, max_layers_y)
        
        print(f"Max layers: {max_layers}, platform range: ({platform_start_x}-{platform_end_x}, {platform_start_y}-{platform_end_y})")
        
        for layer in range(max_layers, 0, -1):
            layer_start_x = platform_start_x - layer * step_width_px
            layer_end_x = platform_end_x + layer * step_width_px
            layer_start_y = platform_start_y - layer * step_width_px
            layer_end_y = platform_end_y + layer * step_width_px
            
            layer_start_x = max(0, layer_start_x)
            layer_end_x = min(terrain.width, layer_end_x)
            layer_start_y = max(0, layer_start_y)
            layer_end_y = min(terrain.length, layer_end_y)
            
            current_height = layer * step_height_px
            
            terrain.height_field_raw[layer_start_x:layer_end_x, layer_start_y:layer_end_y] = current_height
            
            print(f"Layer {layer}: region({layer_start_x}-{layer_end_x}, {layer_start_y}-{layer_end_y}), height={current_height}px ({current_height * terrain.vertical_scale:.3f}m)")
        
        terrain.height_field_raw[platform_start_x:platform_end_x, platform_start_y:platform_end_y] = 0
        
        return terrain

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.02 + 0.14 * difficulty
        discrete_obstacles_height = 0.03 + difficulty * 0.1
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[2]:
            if choice<self.proportions[1]:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[4]:
            if choice<self.proportions[3]:
                step_height *= -1
            self._create_inverted_pyramid_stairs(terrain, step_width=0.744, step_height=0.15, platform_size=3.)
        elif choice < self.proportions[5]:
            num_rectangles = 20
            rectangle_min_size = 0.5
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
            self.add_roughness(terrain)
        elif choice < self.proportions[6]:
            stones_size = 1.5 - 1.2*difficulty
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stones_size, stone_distance=0.1, stone_distance_rand=0, max_height=0.04*difficulty, platform_size=2.)
        elif choice < self.proportions[7]:
            gap_size = random.uniform(self.cfg.gap_size[0], self.cfg.gap_size[1])
            gap_terrain(terrain, gap_size=gap_size, platform_size=3)
            self.add_roughness(terrain)
        elif choice < self.proportions[8]:
            self.add_roughness(terrain)
            pass
        elif choice < self.proportions[9]:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        elif choice < self.proportions[11]:
            self._create_random_height_stairs(terrain, step_width=0.744, min_height=0.0, max_height=0.25, platform_size=3.)
        elif choice < self.proportions[12]:
            if hasattr(self.cfg, 'slope_height'):
                slope_height = random.uniform(self.cfg.slope_height[0], self.cfg.slope_height[1])
            else:
                slope_height = 2.0 + difficulty * 8.0
            
            if hasattr(self.cfg, 'slope_angle'):
                min_angle = self.cfg.slope_angle[0]
                max_angle = self.cfg.slope_angle[1]
                slope_angle = min_angle + difficulty * (max_angle - min_angle)
            else:
                slope_angle = 0.05 + difficulty * 0.47
            
            platform_radius = getattr(self.cfg, 'platform_size', 5.0) / 2.0
            
            ring_slope_terrain(terrain, slope_height=slope_height, slope_angle=slope_angle, platform_radius=platform_radius)
            
            self.add_roughness(terrain, difficulty=0.5)
        else:
            if self.cfg.all_vertical:
                half_slope_difficulty = 1.0
            else:
                difficulty *= 1.3
                if not self.cfg.no_flat:
                    difficulty -= 0.1
                if difficulty > 1:
                    half_slope_difficulty = 1.0
                elif difficulty < 0:
                    self.add_roughness(terrain)
                    terrain.slope_vector = np.array([1, 0., 0]).astype(np.float32)
                    return terrain
                else:
                    half_slope_difficulty = difficulty
            wall_width = 4 - half_slope_difficulty * 4
            if self.cfg.flat_wall:
                half_sloped_terrain(terrain, wall_width=4, start2center=0.5, max_height=0.00)
            else:
                half_sloped_terrain(terrain, wall_width=wall_width, start2center=0.5, max_height=1.5)
            max_height = terrain.height_field_raw.max()
            top_mask = terrain.height_field_raw > max_height - 0.05
            self.add_roughness(terrain, difficulty=1)
            terrain.height_field_raw[top_mask] = max_height
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        if self.cfg.origin_zero_z:
            env_origin_z = 0
        else:
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        # self.env_slope_vec[i, j] = terrain.slope_vector

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def half_sloped_terrain(terrain, wall_width=4, start2center=0.7, max_height=1):
    wall_width_int = max(int(wall_width / terrain.horizontal_scale), 1)
    max_height_int = int(max_height / terrain.vertical_scale)
    slope_start = int(start2center / terrain.horizontal_scale + terrain.length // 2)
    terrain_length = terrain.length
    height2width_ratio = max_height_int / wall_width_int
    xs = np.arange(slope_start, terrain_length)
    heights = (height2width_ratio * (xs - slope_start)).clip(max=max_height_int).astype(np.int16)
    terrain.height_field_raw[slope_start:terrain_length, :] = heights[:, None]
    terrain.slope_vector = np.array([wall_width_int*terrain.horizontal_scale, 0., max_height]).astype(np.float32)
    terrain.slope_vector /= np.linalg.norm(terrain.slope_vector)
    # print(terrain.slope_vector, wall_width)
    # import matplotlib.pyplot as plt
    # plt.imsave('test.png', terrain.height_field_raw, cmap='gray')


def convert_heightfield_to_trimesh_delatin(height_field_raw, horizontal_scale, vertical_scale, max_error=0.01):
    mesh = Delatin(np.flip(height_field_raw, axis=1).T, z_scale=vertical_scale, max_error=max_error)
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2]
    return vertices, mesh.triangles


def ring_slope_terrain(terrain, slope_height=1.0, slope_angle=0.3, platform_radius=4.0):
    """
    Create four-sided ring slope terrain with stairs at edges.
    """
    slope_height_int = int(slope_height / terrain.vertical_scale)
    platform_radius_int = int(platform_radius / terrain.horizontal_scale)
    
    center_x = terrain.length // 2
    center_y = terrain.width // 2
    
    slope_length = slope_height / np.tan(slope_angle)
    slope_end_radius = platform_radius + slope_length
    
    stair_step_width = 0.8
    stair_step_height = 0.12
    
    terrain.height_field_raw.fill(0)
    
    for i in range(terrain.length):
        for j in range(terrain.width):
            dx = (i - center_x) * terrain.horizontal_scale
            dy = (j - center_y) * terrain.horizontal_scale
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance <= platform_radius:
                terrain.height_field_raw[i, j] = 0
                continue
            
            elif distance <= slope_end_radius:
                slope_distance = distance - platform_radius
                height = slope_distance * np.tan(slope_angle)
                
                height_int = int(height / terrain.vertical_scale)
                terrain.height_field_raw[i, j] = height_int
            
            else:
                slope_end_height = slope_length * np.tan(slope_angle)
                
                stair_distance = distance - slope_end_radius
                stair_step_index = int(stair_distance / stair_step_width)
                
                stair_additional_height = stair_step_index * stair_step_height
                total_height = slope_end_height + stair_additional_height
                
                max_total_height = slope_height + 2.5
                total_height = min(total_height, max_total_height)
                
                height_int = int(total_height / terrain.vertical_scale)
                terrain.height_field_raw[i, j] = height_int
    
    terrain.slope_vector = np.array([0., 0., 1.]).astype(np.float32)
