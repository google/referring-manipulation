"""Library for referring manipulation using MDETR + GLIDE."""

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

import os
import pathlib

import lpips
import numpy as np
from numpy import random
from PIL import Image

from third_party.blended_diffusion.CLIP import clip
from third_party.blended_diffusion.optimization import constants
from third_party.blended_diffusion.optimization.augmentations import ImageAugmentations
from third_party.blended_diffusion.optimization.losses import d_clip_loss
from third_party.blended_diffusion.utils import visualization as viz
from third_party.blended_diffusion.utils.metrics_accumulator import MetricsAccumulator
from third_party.glide_text2im.clip.model_creation import create_clip_model
from third_party.glide_text2im.download import load_checkpoint
from third_party.glide_text2im.model_creation import (  # pylint: disable=g-multiple-import
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF


def dilate(image, kernel_size=3):
  kernel = torch.ones(
      1, 1, kernel_size, kernel_size, dtype=image.dtype).to(image.device)
  padding = (kernel_size // 2, kernel_size // 2)
  return torch.clamp(F.conv2d(image, kernel, padding=padding), 0, 1)


class ImageEditorGlide:
  """Main class for Referring Object Manipulation using GLIDE."""

  def __init__(self, args) -> None:
    self.args = args
    os.makedirs(self.args.output_path, exist_ok=True)

    self.ranked_results_path = pathlib.Path(
        os.path.join(self.args.output_path, constants.RANKED_RESULTS_DIR))
    os.makedirs(self.ranked_results_path, exist_ok=True)

    if self.args.export_assets:
      self.assets_path = pathlib.Path(
          os.path.join(self.args.output_path, constants.ASSETS_DIR_NAME))
      os.makedirs(self.assets_path, exist_ok=True)
    if self.args.seed is not None:
      torch.manual_seed(self.args.seed)
      np.random.seed(self.args.seed)
      random.seed(self.args.seed)

    # Create base diffusion model
    self.model_config = model_and_diffusion_defaults()
    self.model_config.update({
        "inpaint": True,
        "diffusion_steps": 1000,
        "timestep_respacing": self.args.timestep_respacing,
        "use_fp16": True,
    })
    self.image_size = (64, 64)
    self.output_image_size = (256, 256)

    self.device = torch.device(
        f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
    print("Using device:", self.device)

    self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
    self.model.requires_grad_(False).eval().to(self.device)
    if self.model_config["use_fp16"]:
      self.model.convert_to_fp16()
    self.model.load_state_dict(load_checkpoint("base-inpaint", self.device))
    print("total base parameters",
          sum(x.numel() for x in self.model.parameters()))

    # Create upsampler model.
    options_up = model_and_diffusion_defaults_upsampler()
    options_up["inpaint"] = True
    options_up["use_fp16"] = True
    # use 27 diffusion steps for very fast sampling
    options_up["timestep_respacing"] = "fast27"
    self.options_up = options_up
    self.model_up, self.diffusion_up = create_model_and_diffusion(**options_up)
    self.model_up.requires_grad_(False).eval().to(self.device)
    if options_up["use_fp16"]:
      self.model_up.convert_to_fp16()
    self.model_up.load_state_dict(
        load_checkpoint("upsample-inpaint", self.device))
    print("total upsampler parameters",
          sum(x.numel() for x in self.model_up.parameters()))

    # Load ReferSeg model
    self.model_seg = torch.hub.load("ashkamath/mdetr:main",
                                    "mdetr_efficientnetB3_phrasecut",
                                    pretrained=True,
                                    return_postprocessor=False)
    self.model_seg = self.model_seg.to(self.device)
    self.model_seg.eval()

    # Load (original, unnoised) CLIP model
    self.clip_model = (
        clip.load("ViT-B/16",
                  device=self.device,
                  jit=False)[0].eval().requires_grad_(False)
    )
    self.clip_size = self.clip_model.visual.input_resolution
    self.clip_normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    # Create (noised) CLIP model.
    self.clip_model_glide = create_clip_model(device=self.device)
    self.clip_model_glide.image_encoder.load_state_dict(
        load_checkpoint("clip/image-enc", self.device))
    self.clip_model_glide.text_encoder.load_state_dict(
        load_checkpoint("clip/text-enc", self.device))
    self.clip_size_glide = 64

    # Load LPIPS model
    self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)

    self.image_augmentations = ImageAugmentations(self.clip_size,
                                                  self.args.aug_num)
    self.metrics_accumulator = MetricsAccumulator()
    self.seg_transform = transforms.Compose([
        transforms.Resize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

  def unscale_timestep(self, t):
    unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()
    return unscaled_timestep

  def unaugmented_clip_distance(self, x, text_embed):
    x = TF.resize(x, [self.clip_size, self.clip_size])
    image_embeds = self.clip_model.encode_image(x).float()
    dists = d_clip_loss(image_embeds, text_embed)

    return dists.item()

  def local_clip_distance(self, x, text_embed):
    """Calculate localized rankings with bbox context."""
    x1, y1, x2, y2 = self.bboxes_scaled
    x = x[:, :, y1:y2 + 1, x1:x2 + 1]
    x = TF.resize(x, [self.clip_size, self.clip_size])
    image_embeds = self.clip_model.encode_image(x).float()
    dists = d_clip_loss(image_embeds, text_embed)

    return dists.item()

  def update_args(self, args):
    """Overwrites self.args with given args."""
    self.args = args
    os.makedirs(self.args.output_path, exist_ok=True)

    self.ranked_results_path = pathlib.Path(
        os.path.join(self.args.output_path, constants.RANKED_RESULTS_DIR))
    os.makedirs(self.ranked_results_path, exist_ok=True)

    if self.args.export_assets:
      self.assets_path = pathlib.Path(
          os.path.join(self.args.output_path, constants.ASSETS_DIR_NAME))
      os.makedirs(self.assets_path, exist_ok=True)

  def read_image(self):
    """Read and preprocess the input image."""
    init_image_pil = Image.open(self.args.init_image).convert("RGB")
    # Build input for segmentation model before resizing
    self.seg_input = self.seg_transform(init_image_pil).unsqueeze(0).to(
        self.device)

    def process_image(pil_img, size):
      pil_img = pil_img.resize(size, resample=Image.BICUBIC)
      img = np.array(pil_img)
      img_tensor = torch.from_numpy(img)[None].permute(
          0, 3, 1, 2).float() / 127.5 - 1
      return img_tensor.to(self.device)

    self.init_image_64 = process_image(init_image_pil, size=self.image_size)
    self.init_image_256 = process_image(
        init_image_pil, size=self.output_image_size)
    self.init_image_pil = init_image_pil.resize(
        self.output_image_size, resample=Image.BICUBIC)

  def edit_image_by_prompt(self):
    """Main script for text-conditioned editing."""
    # Read input images - sets `self.seg_input` and `self.init_image`
    self.read_image()

    output = self.run_refer_segmentation()
    self.masked_image_256, self.masked_image_64, self.mask_256, self.mask_64 = self.build_glide_input(
        output, mode="seg", dilate_kernel=self.args.dilate_kernel)
    upsample_temp = 0.997
    use_clip = not self.args.cfg
    samples = self.run_glide_base_64(
        self.args.guidance_scale, use_clip=use_clip)
    up_samples = self.run_glide_upsample(samples, upsample_temp)

    # Save some variables for visualization
    #   mask representation is opposite from GLIDE vs blended_diffusion
    self.mask_pil = TF.to_pil_image(1 - self.mask_256.squeeze(0))
    if self.args.export_assets:
      self.init_image_pil.save(self.assets_path / "input_image.png")
      self.mask_pil.save(self.assets_path / "mask.png")
      TF.to_pil_image(
          self.masked_image_256.squeeze(0).add(1).div(2).clamp(0, 1)).save(
              self.assets_path / "masked_image.png")
      viz.plot_results(self.init_image_pil, self.score, self.bboxes_256,
                       self.assets_path / "masked_image_color.png",
                       1 - self.mask_256.squeeze(0))

    # Calculated text_embed for clip loss
    text_embed = self.clip_model.encode_text(
        clip.tokenize(self.args.prompt).to(self.device)
    ).float()
    for b in range(self.args.batch_size):
      pred_image = up_samples[b]
      visualization_path = pathlib.Path(
          os.path.join(self.args.output_path, self.args.output_file)
      )
      visualization_path = visualization_path.with_name(
          f"{visualization_path.stem}_b_{b}{visualization_path.suffix}"
      )

      pred_image = pred_image.add(1).div(2).clamp(0, 1)
      pred_image_pil = TF.to_pil_image(pred_image)
      masked_pred_image = (1 - self.mask_256) * pred_image.unsqueeze(0)
      if self.args.localized_rank:
        final_distance = self.local_clip_distance(
            pred_image.unsqueeze(0), text_embed
        )
      else:
        final_distance = self.unaugmented_clip_distance(
            masked_pred_image, text_embed
        )
      formatted_distance = f"{final_distance:.4f}"

      if self.args.export_assets:
        path_friendly_distance = formatted_distance.replace(".", "")
        ranked_pred_path = self.ranked_results_path / (
            path_friendly_distance + "_" + visualization_path.name)
        pred_image_pil.save(ranked_pred_path)

      viz.show_edited_masked_image(
          refer_prompt=self.args.refer_prompt,
          target_prompt=self.args.prompt,
          source_image=self.init_image_pil,
          edited_image=pred_image_pil,
          mask=self.mask_pil,
          path=visualization_path,
          distance=formatted_distance,
      )

  def run_glide_base_64(self, guidance_scale, use_clip=False):
    """GLIDE inpainting model inference using difference guidance schemes."""
    refer_prompt = self.args.refer_prompt
    prompt = self.args.prompt
    options = self.model_config
    batch_size = self.args.batch_size

    # Create the text tokens to feed to the model.
    tokens = self.model.tokenizer.encode(prompt)
    tokens, mask = self.model.tokenizer.padded_tokens_and_mask(
        tokens, options["text_ctx"])

    # Create the classifier-free guidance tokens (empty)
    if not use_clip:
      full_batch_size = batch_size * 2
      if self.args.ccfg:
        refer_tokens = self.model.tokenizer.encode(refer_prompt)
        uncond_tokens, uncond_mask = self.model.tokenizer.padded_tokens_and_mask(
            refer_tokens, options["text_ctx"])
        if not self.args.prompt:  # self.args.prompt == ""
          # It's better to start from a masked image for inpainting
          inpaint_image = self.masked_image_64.repeat(full_batch_size, 1, 1,
                                                      1).to(self.device)
        else:
          # Better start from the original image for editing
          inpaint_image = self.init_image_64.repeat(full_batch_size, 1, 1,
                                                    1).to(self.device)
        print("using ccfg")
      else:
        uncond_tokens, uncond_mask = self.model.tokenizer.padded_tokens_and_mask(
            [], options["text_ctx"])
        inpaint_image = self.masked_image_64.repeat(full_batch_size, 1, 1,
                                                    1).to(self.device)
        print("using cfg")

      # Pack the tokens together into model kwargs.
      model_kwargs = dict(
          tokens=torch.tensor(
              [tokens] * batch_size + [uncond_tokens] * batch_size,
              device=self.device),
          mask=torch.tensor(
              [mask] * batch_size + [uncond_mask] * batch_size,
              dtype=torch.bool,
              device=self.device,
          ),
          # Inpainting image and mask
          inpaint_image=inpaint_image,
          inpaint_mask=self.mask_64.repeat(full_batch_size, 1, 1, 1).to(
              self.device),
      )
    else:
      # Pack the tokens together into model kwargs.
      model_kwargs = dict(
          tokens=torch.tensor([tokens] * batch_size, device=self.device),
          mask=torch.tensor(
              [mask] * batch_size, dtype=torch.bool, device=self.device),
          # Masked inpainting image
          inpaint_image=self.masked_image_64.repeat(batch_size, 1, 1, 1).to(
              self.device),
          inpaint_mask=self.mask_64.repeat(batch_size, 1, 1, 1).to(self.device),
      )

    # Create an classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
      half = x_t[:len(x_t) // 2]
      combined = torch.cat([half, half], dim=0)
      model_out = self.model(combined, ts, **kwargs)
      eps, rest = model_out[:, :3], model_out[:, 3:]
      cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
      half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
      eps = torch.cat([half_eps, half_eps], dim=0)
      return torch.cat([eps, rest], dim=1)

    def denoised_fn(x_start):
      # Force the model to have the exact right x_start predictions
      # for the part of the image which is known.
      return (x_start * (1 - model_kwargs["inpaint_mask"]) +
              model_kwargs["inpaint_image"] * model_kwargs["inpaint_mask"])

    # Setup guidance function for CLIP model.
    if use_clip:
      cond_fn = self.clip_model_glide.cond_fn([prompt] * batch_size,
                                              guidance_scale,
                                              [refer_prompt] * batch_size)
    else:
      cond_fn = None

    # Sample from the base model.
    self.model.del_cache()
    if not use_clip:
      samples = self.diffusion.p_sample_loop(
          model_fn,
          (full_batch_size, 3, options["image_size"], options["image_size"]),
          device=self.device,
          clip_denoised=True,
          progress=True,
          model_kwargs=model_kwargs,
          cond_fn=cond_fn,
          denoised_fn=denoised_fn,
      )[:batch_size]
    else:
      samples = self.diffusion.p_sample_loop(
          self.model,
          (batch_size, 3, options["image_size"], options["image_size"]),
          device=self.device,
          clip_denoised=True,
          progress=True,
          model_kwargs=model_kwargs,
          cond_fn=cond_fn,
          denoised_fn=denoised_fn,
      )
    self.model.del_cache()

    return samples

  def run_glide_upsample(self, samples, upsample_temp):
    """GLIDE x4 upsampling model inference."""
    prompt = self.args.prompt
    batch_size = self.args.batch_size
    options_up = self.options_up

    tokens = self.model_up.tokenizer.encode(prompt)
    tokens, mask = self.model_up.tokenizer.padded_tokens_and_mask(
        tokens, options_up["text_ctx"])

    # Create the model conditioning dict.
    model_kwargs = dict(
        # Low-res image to upsample.
        low_res=((samples+1)*127.5).round()/127.5 - 1,

        # Text tokens
        tokens=torch.tensor([tokens] * batch_size, device=self.device),
        mask=torch.tensor(
            [mask] * batch_size,
            dtype=torch.bool,
            device=self.device,
        ),

        # Masked inpainting image.
        inpaint_image=self.masked_image_256.repeat(batch_size, 1, 1,
                                                   1).to(self.device),
        inpaint_mask=self.mask_256.repeat(batch_size, 1, 1, 1).to(self.device),
    )

    def denoised_fn(x_start):
      # Force the model to have the exact right x_start predictions
      # for the part of the image which is known.
      return (x_start * (1 - model_kwargs["inpaint_mask"]) +
              model_kwargs["inpaint_image"] * model_kwargs["inpaint_mask"])

    # Sample from the base model.
    self.model_up.del_cache()
    up_shape = (batch_size, 3, options_up["image_size"],
                options_up["image_size"])
    up_samples = self.diffusion_up.p_sample_loop(
        self.model_up,
        up_shape,
        noise=torch.randn(up_shape, device=self.device) * upsample_temp,
        device=self.device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
        denoised_fn=denoised_fn,
    )[:batch_size]
    self.model_up.del_cache()

    return up_samples

  def build_glide_input(self, out_mdetr, mode="seg", dilate_kernel=3):
    """Prepare MDETR outputs to match the input formats of GLIDE."""
    # mode: "bbox" or "seg"
    probas = 1 - out_mdetr["pred_logits"].softmax(-1)[0, :, -1]

    # Just keep the one with the maximum score
    keep = torch.argmax(probas)
    bboxes_scaled = viz.rescale_bboxes(
        out_mdetr["pred_boxes"][0, keep:keep + 1], (64, 64))

    bboxes_enlarged = viz.enlarge_bbox(
        out_mdetr["pred_boxes"][0, keep:keep + 1], scale=1.3,
        size=(256, 256)).squeeze()
    self.bboxes_scaled = (int(bboxes_enlarged[0]), int(bboxes_enlarged[1]),
                          int(bboxes_enlarged[2]), int(bboxes_enlarged[3]))

    # For visualization
    if self.args.export_assets:
      bboxes_256 = viz.rescale_bboxes(out_mdetr["pred_boxes"][0, keep:keep + 1],
                                      (256, 256)).squeeze()
      self.bboxes_256 = (int(bboxes_256[0]), int(bboxes_256[1]),
                         int(bboxes_256[2]), int(bboxes_256[3]))
      self.score = probas[keep:keep + 1]

    mask_64 = torch.ones_like(self.init_image_64[:, :1])
    bboxes_scaled = bboxes_scaled.squeeze()

    if mode == "bbox":
      x1, y1, x2, y2 = int(bboxes_scaled[0]), int(bboxes_scaled[1]), int(
          bboxes_scaled[2]), int(bboxes_scaled[3])
      mask_64[:, :, y1:y2 + 1, x1:x2 + 1] = 0
    elif mode == "seg":
      masks_64 = F.interpolate(
          out_mdetr["pred_masks"],
          size=(64, 64),
          mode="bilinear",
          align_corners=False)
      mask_64 = (masks_64[0, keep:keep+1].sigmoid() > 0.5).to(torch.float)
      mask_64 = 1 - dilate(mask_64.unsqueeze(0), dilate_kernel)
    mask_256 = F.interpolate(mask_64, size=(256, 256), mode="nearest")

    print(self.init_image_256.device, mask_256.device)
    masked_image_256 = self.init_image_256 * mask_256
    masked_image_64 = self.init_image_64 * mask_64

    # convert mask to float type
    im_dtype = self.init_image_256.dtype
    mask_256, mask_64 = mask_256.to(im_dtype), mask_64.to(im_dtype)
    return masked_image_256, masked_image_64, mask_256, mask_64

  @torch.no_grad()
  def run_refer_segmentation(self):
    """Run MDETR referring image segmentation model."""
    caption = self.args.refer_prompt
    img = self.seg_input

    # propagate through the model
    outputs = self.model_seg(img, [caption])

    return outputs
