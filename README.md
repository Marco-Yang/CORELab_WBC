## Setup
```bash
conda create -n WBCRL python=3.8
conda activate WBCRL

cd WBC_RL

pip install torch torchvision torchaudio

cd third_party/isaacgym/python && pip install -e .

cd ../..
cd rsl_rl && pip install -e .

cd ..
cd skrl && pip install -e .

cd ../..
cd low-level && pip install -e .

pip install numpy pydelatin tqdm imageio-ffmpeg opencv-python wandb
```
## Pretrained model
please replace the folder low-level/logs with the folder shown in the link
https://drive.google.com/file/d/1YC3OAFx--k47YYFHIeJ6pHEbad53er3q/view?usp=drive_link

## Train
```bash
cd low-level/legged_gym/scripts
python train.py --headless --exptid SOME_YOUR_DESCRIPTION --proj_name b1z1-low --task b1z1 --sim_device cuda:0 --rl_device cuda:0 --observe_gait_commands

## train_stairs
python legged_gym/scripts/train.py --task b1z1_stairs_simple --headless --exptid stairs_train1 --proj_name b1z1-training --sim_device cuda:1 --rl_device cuda:1 --num_envs 2048 --max_iterations 80000 --observe_gait_commands

## train_stairs
## continue training based on the previous pretrained model
cd low-level && python ./legged_gym/scripts/train_continue.py --pretrained_path ./logs/b1z1-low/model_38000/model_38000.pt --exptid stairs_random --task b1z1_stairs_simple  --max_iterations 80000 --num_envs 2048 --headless

## train_ring_slope
# continue training based on the previous pretrained model
cd low-level && __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia PYOPENGL_PLATFORM=egl \
python legged_gym/scripts/train_continue.py \
  --task b1z1_ring_slope \
  --pretrained_path logs/b1z1-stairs-continued/stairs_18cm_downstairs_stability/model_44000.pt \
  --exptid slope_transfer_from_stairs \
  --proj_name b1z1-ring-slope \
  --num_envs 1024 \
  --start_iteration 44000 \
  --headless 


```

##  Play
Only need to specify `--exptid`. The parser will automatically find corresponding runs.
Assuming the current path is ~/CORELab_WBC

```bash

## flat and rough ground
cd low-level && LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia PYOPENGL_PLATFORM=egl python legged_gym/scripts/play.py   --exptid stairs_18cm_downstairs_stability   --task b1z1   --proj_name b1z1-stairs-continued   --checkpoint 44000   --observe_gait_commands

## stairs with regular_heights
cd low-level && LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia PYOPENGL_PLATFORM=egl python legged_gym/scripts/play.py   --exptid stairs_18cm_downstairs_stability   --task b1z1_stairs_simple   --proj_name b1z1-stairs-continued   --checkpoint 44000   --observe_gait_commands

## stairs with random_heights
cd low-level && LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia PYOPENGL_PLATFORM=egl python legged_gym/scripts/play.py   --exptid stairs_18cm_downstairs_stability   --task b1z1_stairs_random   --proj_name b1z1-stairs-continued   --checkpoint 44000   --observe_gait_commands

## ring slope env
cd low-level && LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia PYOPENGL_PLATFORM=egl python legged_gym/scripts/play.py   --exptid slope_transfer_from_stairs   --task b1z1_ring_slope   --proj_name b1z1-ring-slope   --checkpoint 45000   --observe_gait_commands
```
