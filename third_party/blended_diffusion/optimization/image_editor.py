"""Library for referring manipulation using MDETR + Blended-diffusion."""

import os
import pathlib

import lpips
import numpy as np
from numpy import random
from PIL import Image

from third_party.blended_diffusion.CLIP import clip
from third_party.blended_diffusion.guided_diffusion.guided_diffusion.script_util import (  # pylint: disable=g-multiple-import
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from third_party.blended_diffusion.optimization import constants
from third_party.blended_diffusion.optimization.augmentations import ImageAugmentations
from third_party.blended_diffusion.optimization.losses import d_clip_loss
from third_party.blended_diffusion.optimization.losses import range_loss
from third_party.blended_diffusion.utils import visualization as viz
from third_party.blended_diffusion.utils.metrics_accumulator import MetricsAccumulator
from third_party.blended_diffusion.utils.video import save_video

import torch
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from torchvision import transforms
from torchvision.transforms import functional as TF


def dilate(image, kernel_size=3):
  kernel = torch.ones(
      1, 1, kernel_size, kernel_size, dtype=image.dtype).to(image.device)
  padding = (kernel_size // 2, kernel_size // 2)
  return torch.clamp(F.conv2d(image, kernel, padding=padding), 0, 1)


class ImageEditor:
  """Main class for Referring Object Manipulation using blended-diffusion."""

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

    self.model_config = model_and_diffusion_defaults()
    self.model_config.update({
        "attention_resolutions": "32, 16, 8",
        "class_cond": self.args.model_output_size == 512,
        "diffusion_steps": 1000,
        "rescale_timesteps": True,
        "timestep_respacing": self.args.timestep_respacing,
        "image_size": self.args.model_output_size,
        "learn_sigma": True,
        "noise_schedule": "linear",
        "num_channels": 256,
        "num_head_channels": 64,
        "num_res_blocks": 2,
        "resblock_updown": True,
        "use_fp16": True,
        "use_scale_shift_norm": True,
    })
    self.image_size = (self.model_config["image_size"],
                       self.model_config["image_size"])

    self.device = torch.device(
        f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print("Using device:", self.device)

    # Load diffusion model
    self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
    self.model.load_state_dict(
        torch.load(
            "checkpoints/256x256_diffusion_uncond.pt"
            if self.args.model_output_size == 256 else
            "checkpoints/512x512_diffusion.pt",
            map_location="cpu",
        ))
    self.model.requires_grad_(False).eval().to(self.device)
    for name, param in self.model.named_parameters():
      if "qkv" in name or "norm" in name or "proj" in name:
        param.requires_grad_()
    if self.model_config["use_fp16"]:
      self.model.convert_to_fp16()

    # Load ReferSeg model
    self.model_seg = torch.hub.load(
        "ashkamath/mdetr:main",
        "mdetr_efficientnetB3_phrasecut",
        pretrained=True,
        return_postprocessor=False)
    self.model_seg = self.model_seg.to(self.device)
    self.model_seg.eval()

    # Load CLIP model
    self.clip_model = (
        clip.load("ViT-B/16", device=self.device,
                  jit=False)[0].eval().requires_grad_(False))
    self.clip_size = self.clip_model.visual.input_resolution
    self.clip_normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711])

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

    # Read input images - sets `self.seg_input` and `self.init_image`
    self.read_image()

  def unscale_timestep(self, t):
    unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()
    return unscaled_timestep

  def clip_loss(self, x_in, text_embed):
    """Calculates CLIP loss for the masked regions for ranking."""
    clip_loss = torch.tensor(0)

    if self.mask is not None:
      masked_input = x_in * self.mask
    else:
      masked_input = x_in
    augmented_input = self.image_augmentations(masked_input).add(1).div(2)
    clip_in = self.clip_normalize(augmented_input)
    image_embeds = self.clip_model.encode_image(clip_in).float()
    dists = d_clip_loss(image_embeds, text_embed)

    # We want to sum over the averages
    for i in range(self.args.batch_size):
      # We want to average at the "augmentations level"
      clip_loss = clip_loss + dists[i::self.args.batch_size].mean()

    return clip_loss

  def unaugmented_clip_distance(self, x, text_embed):
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
    self.init_image_pil = Image.open(self.args.init_image).convert("RGB")
    # Build input for segmentation model before resizing
    self.seg_input = self.seg_transform(self.init_image_pil).unsqueeze(0).to(
        self.device)
    # Resize to output image size
    self.init_image_pil = self.init_image_pil.resize(
        self.image_size, resample=Image.LANCZOS)
    self.init_image = (
        TF.to_tensor(self.init_image_pil).to(
            self.device).unsqueeze(0).mul(2).sub(1))

  def edit_image_by_prompt(self):
    """Main script for text-conditioned editing."""
    # Read input images - sets `self.seg_input` and `self.init_image`
    self.read_image()
    self.run_refer_segmentation()

    text_embed = self.clip_model.encode_text(
        clip.tokenize(self.args.prompt).to(self.device)
    ).float()

    self.mask_pil = TF.to_pil_image(self.mask.squeeze(0))
    if self.args.export_assets:
      mask_path = self.assets_path / pathlib.Path(
          self.args.output_file.replace(".png", "_mask.png"))
      self.mask_pil.save(mask_path)

    def cond_fn(x, t, y=None):
      if not self.args.prompt:  # self.args.prompt == ""
        return torch.zeros_like(x)

      with torch.enable_grad():
        x = x.detach().requires_grad_()
        t = self.unscale_timestep(t)

        out = self.diffusion.p_mean_variance(
            self.model, x, t, clip_denoised=False, model_kwargs={"y": y})

        fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
        x_in = out["pred_xstart"] * fac + x * (1 - fac)

        loss = torch.tensor(0)
        if self.args.clip_guidance_lambda != 0:
          clip_loss = self.clip_loss(
              x_in, text_embed) * self.args.clip_guidance_lambda
          loss = loss + clip_loss
          self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())

        if self.args.range_lambda != 0:
          r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
          loss = loss + r_loss
          self.metrics_accumulator.update_metric("range_loss", r_loss.item())

        if self.args.background_preservation_loss:
          if self.mask is not None:
            masked_background = x_in * (1 - self.mask)
          else:
            masked_background = x_in

          if self.args.lpips_sim_lambda:
            loss = (
                loss +
                self.lpips_model(masked_background, self.init_image).sum() *
                self.args.lpips_sim_lambda)
          if self.args.l2_sim_lambda:
            loss = (
                loss + mse_loss(masked_background, self.init_image) *
                self.args.l2_sim_lambda)

        return -torch.autograd.grad(loss, x)[0]

    @torch.no_grad()
    def postprocess_fn(out, t):
      if self.mask is not None:
        background_stage_t = self.diffusion.q_sample(self.init_image, t[0])
        background_stage_t = torch.tile(
            background_stage_t, dims=(self.args.batch_size, 1, 1, 1))
        out["sample"] = out["sample"] * self.mask + background_stage_t * (
            1 - self.mask)

      return out

    save_image_interval = self.diffusion.num_timesteps // 5
    for iteration_number in range(self.args.iterations_num):
      print(f"Start iterations {iteration_number}")

      model_kwargs = {}
      if self.args.model_output_size != 256:
        model_kwargs = {
            "y":
                torch.zeros([self.args.batch_size],
                            device=self.device,
                            dtype=torch.long)
        }
      samples = self.diffusion.p_sample_loop_progressive(
          self.model,
          (
              self.args.batch_size,
              3,
              self.model_config["image_size"],
              self.model_config["image_size"],
          ),
          clip_denoised=False,
          model_kwargs=model_kwargs,
          cond_fn=cond_fn,
          progress=True,
          skip_timesteps=self.args.skip_timesteps,
          init_image=self.init_image,
          postprocess_fn=None
          if self.args.local_clip_guided_diffusion else postprocess_fn,
          randomize_class=True,
      )

      intermediate_samples = [[] for i in range(self.args.batch_size)]
      total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1
      for j, sample in enumerate(samples):
        should_save_image = j % save_image_interval == 0 or j == total_steps
        if should_save_image or self.args.save_video:
          self.metrics_accumulator.print_average_metric()

          for b in range(self.args.batch_size):
            pred_image = sample["pred_xstart"][b]
            visualization_path = pathlib.Path(
                os.path.join(self.args.output_path, self.args.output_file))
            visualization_path = visualization_path.with_name(
                f"{visualization_path.stem}_i_{iteration_number}_b_{b}{visualization_path.suffix}"
            )

            if (self.mask is not None and self.args.enforce_background and
                j == total_steps and not self.args.local_clip_guided_diffusion):
              pred_image = (
                  self.init_image[0] * (1 - self.mask[0]) +
                  pred_image * self.mask[0])
            pred_image = pred_image.add(1).div(2).clamp(0, 1)
            pred_image_pil = TF.to_pil_image(pred_image)
            masked_pred_image = self.mask * pred_image.unsqueeze(0)
            final_distance = self.unaugmented_clip_distance(
                masked_pred_image, text_embed)
            formatted_distance = f"{final_distance:.4f}"

            if self.args.export_assets:
              pred_path = self.assets_path / visualization_path.name
              pred_image_pil.save(pred_path)

            if j == total_steps:
              path_friendly_distance = formatted_distance.replace(".", "")
              ranked_pred_path = self.ranked_results_path / (
                  path_friendly_distance + "_" + visualization_path.name)
              pred_image_pil.save(ranked_pred_path)

            intermediate_samples[b].append(pred_image_pil)
            if should_save_image:
              viz.show_edited_masked_image(
                  refer_prompt=self.args.refer_prompt,
                  target_prompt=self.args.prompt,
                  source_image=self.init_image_pil,
                  edited_image=pred_image_pil,
                  mask=self.mask_pil,
                  path=visualization_path,
                  distance=formatted_distance,
              )

      if self.args.save_video:
        for b in range(self.args.batch_size):
          video_name = self.args.output_file.replace(
              ".png", f"_i_{iteration_number}_b_{b}.avi")
          video_path = os.path.join(self.args.output_path, video_name)
          save_video(intermediate_samples[b], video_path)

  @torch.no_grad()
  def run_refer_segmentation(self, mode="seg", dilate_kernel=3):
    """Run MDETR referring image segmentation model."""
    caption = self.args.refer_prompt

    # mean-std normalize the input image (batch-size: 1)
    img = self.seg_input

    # propagate through the model
    outputs = self.model_seg(img, [caption])

    # keep only predictions with 0.9+ confidence
    probas = 1 - outputs["pred_logits"].softmax(-1)[0, :, -1]

    # Just keep the one with the maximum score
    keep = torch.argmax(probas)
    # bboxes_scaled = viz.rescale_bboxes(
    #     outputs["pred_boxes"][0, keep:keep + 1], (64, 64))

    if mode == "bbox":
      # x1, y1, x2, y2 = int(bboxes_scaled[0]), int(bboxes_scaled[1]), int(
      #     bboxes_scaled[2]), int(bboxes_scaled[3])
      # mask_256[:, :, y1:y2 + 1, x1:x2 + 1] = 0
      raise NotImplementedError("BBox mode not implemented for Blended.")
    elif mode == "seg":
      masks_256 = F.interpolate(
          outputs["pred_masks"],
          size=self.image_size,
          mode="bilinear",
          align_corners=False)
      mask_256 = (masks_256[0, keep:keep + 1].sigmoid() > 0.5).to(torch.float)
      mask_256 = dilate(mask_256.unsqueeze(0), dilate_kernel)
      if self.args.invert_mask:  # Actually, this is a true mask
        mask_256 = 1 - mask_256

    self.mask = mask_256.to(self.init_image.dtype)

    return outputs
