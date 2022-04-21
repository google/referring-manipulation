#!/bin/bash

# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

image_id=2351408
refer_prompt="left hand on girl"
target_prompt="a boxing glove"
save_dir="hand2boxing"
input_image="./dataset/vg_images/${image_id}.jpg"
output_path="outputs/${image_id}_${save_dir}/"
seed=1
gpu=0

# GLIDE + CCFG
CUDA_VISIBLE_DEVICES=${gpu} python main.py \
	-rp "${refer_prompt}" \
	-p  "${target_prompt}" \
	-i "${input_image}" \
	--seed ${seed} \
	--guidance_scale 15 \
	--glide \
	--batch_size 24 \
	--timestep_respacing 100 \
	--output_path "${output_path}glide_ccfg" \
	--ccfg \
	--localized_rank \
	--export_assets 

# GLIDE + CFG
CUDA_VISIBLE_DEVICES=${gpu} python main.py \
	-rp "${refer_prompt}" \
	-p  "${target_prompt}" \
	-i "${input_image}" \
	--seed ${seed} \
	--glide \
	--batch_size 24 \
	--timestep_respacing 100 \
	--output_path "${output_path}glide_cfg" \
	--cfg \
	--localized_rank \
	--export_assets 

# GLIDE + CLIP
CUDA_VISIBLE_DEVICES=${gpu} python main.py \
	-rp "${refer_prompt}" \
	-p  "${target_prompt}" \
	-i "${input_image}" \
	--seed ${seed} \
	--glide \
	--batch_size 24 \
	--timestep_respacing 100 \
	--guidance_scale 2 \
	--output_path "${output_path}glide_clip" \
	--localized_rank \
	--export_assets 

# Blended diffusion
CUDA_VISIBLE_DEVICES=${gpu} python main.py \
	-rp "${refer_prompt}" \
	-p  "${target_prompt}" \
	-i "${input_image}" \
	--seed ${seed} \
	--batch_size 3 \
	--aug_num 8 \
	--timestep_respacing 100 \
	--output_path "${output_path}blend" \
	--export_assets 