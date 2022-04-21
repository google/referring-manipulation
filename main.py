"""Main entry point for running Referring Object Manipulation models."""

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

import json
import random

from third_party.blended_diffusion.optimization.arguments import get_arguments
from third_party.blended_diffusion.optimization.image_editor import ImageEditor
from image_editor_glide import ImageEditorGlide


if __name__ == "__main__":
  args = get_arguments()
  if args.ccfg:
    args.cfg = True

  if args.vg:  # Set False by default - this section is used for inpainting-only
    # Random source image from Visual Genomes Dataset
    with open(
        "dataset/mdetr_annotations/finetune_phrasecut_test.json", "r") as f:
      dataset = json.load(f)
    im_root = "dataset/vg_images/"
    imglist = dataset["images"]
    print("Total number of data: %d" % len(imglist))
    k = args.num_samples
    random.seed(args.seed)
    sampled_data = random.sample(imglist, k=k)

    if args.glide:
      args.output_path = "outputs/glide"
      image_editor = ImageEditorGlide(args)
    else:
      args.output_path = "outputs/blend"
      image_editor = ImageEditor(args)
    out_path = args.output_path

    for sample in sampled_data:
      sampled_image = im_root + sample["file_name"]
      sampled_caption = sample["caption"]
      print("Sampled image: %s" % sampled_image)
      print("Sampled caption: %s" % sampled_caption)
      args.refer_prompt = sampled_caption
      args.prompt = ""
      args.init_image = sampled_image
      args.output_path = "%s/%s_%s_i%04d" % (out_path, sample["file_name"][:-4],
                                             sampled_caption.replace(" ", "_"),
                                             int(args.timestep_respacing))
      image_editor.update_args(args)
      image_editor.edit_image_by_prompt()

  elif args.glide:  # Default entry point
    print("Source image: %s" % args.init_image)
    print("Referring prompt: %s" % args.refer_prompt)
    print("Target prompt: %s" % args.prompt)
    image_editor = ImageEditorGlide(args)
    image_editor.edit_image_by_prompt()

  else:
    image_editor = ImageEditor(args)
    image_editor.edit_image_by_prompt()
