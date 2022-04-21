import argparse


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--glide", help="Indicator for using GLIDE", action="store_true", dest="glide",
    )
    
    parser.add_argument(
        "--vg", help="Indicator for batch inference on Visual Genome images", action="store_true", dest="vg",
    )

    # Inputs
    parser.add_argument(
        "-rp", "--refer_prompt", type=str, help="The prompt for the referring the editing region", required=True
    )
    parser.add_argument(
        "-p", "--prompt", type=str, help="The prompt for the desired editing", required=True
    )
    parser.add_argument(
        "-i", "--init_image", type=str, help="The path to the source image input", required=False
    )
    parser.add_argument("--mask", type=str, help="The path to the mask to edit with", default=None)
    parser.add_argument(
        "--num_samples",
        type=int,
        help="The number of samples to choose from visual-genomes images",
        default=10,
    )

    # Diffusion
    parser.add_argument(
        "--skip_timesteps",
        type=int,
        help="How many steps to skip during the diffusion.",
        default=25,
    )
    parser.add_argument(
        "--local_clip_guided_diffusion",
        help="Indicator for using local CLIP guided diffusion (for baseline comparison)",
        action="store_true",
        dest="local_clip_guided_diffusion",
    )

    # For more details read guided-diffusion/guided_diffusion/respace.py
    parser.add_argument(
        "--timestep_respacing",
        type=str,
        help="How to respace the intervals of the diffusion process (number between 1 and 1000).",
        default="1000",
    )
    parser.add_argument(
        "--model_output_size",
        type=int,
        help="The resolution of the outputs of the diffusion model",
        default=256,
        choices=[256, 512],
    )

    # Augmentations
    parser.add_argument("--aug_num", type=int, help="The number of augmentation", default=8)
    parser.add_argument("--dilate_kernel", type=int, help="The kernel size for mask dilation", default=3)

    # Loss
    parser.add_argument(
        "--guidance_scale",
        type=float,
        help="Controls how much the image should look like the prompt",
        default=5,
    )
    parser.add_argument(
        "--clip_guidance_lambda",
        type=float,
        help="Controls how much the image should look like the prompt",
        default=1000,
    )
    parser.add_argument(
        "--range_lambda",
        type=float,
        help="Controls how far out of range RGB values are allowed to be",
        default=50,
    )
    parser.add_argument(
        "--lpips_sim_lambda",
        type=float,
        help="The LPIPS similarity to the input image",
        default=1000,
    )
    parser.add_argument(
        "--l2_sim_lambda", type=float, help="The L2 similarity to the input image", default=10000,
    )
    parser.add_argument(
        "--background_preservation_loss",
        help="Indicator for using the background preservation loss",
        action="store_true",
    )
    parser.add_argument(
        "--cfg",
        help="Indicator for using classifier-free guidance (if not, use CLIP)",
        action="store_true",
    )
    parser.add_argument(
        "--ccfg",
        help="Indicator for using conditional classifier-free guidance",
        action="store_true",
    )
    parser.add_argument(
        "--localized_rank",
        help="Indicator for using localized CLIP ranking",
        action="store_true",
    )

    # Mask
    parser.add_argument(
        "--invert_mask",
        help="Indicator for mask inversion",
        action="store_true",
        dest="invert_mask",
    )
    parser.add_argument(
        "--no_enforce_background",
        help="Indicator disabling the last background enforcement",
        action="store_false",
        dest="enforce_background",
    )

    # Misc
    parser.add_argument("--seed", type=int, help="The random seed", default=404)
    parser.add_argument("--gpu_id", type=int, help="The GPU ID", default=0)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="The filename to save, must be png",
        default="output.png",
    )
    parser.add_argument("--iterations_num", type=int, help="The number of iterations", default=8)
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The number number if images to sample each diffusion process",
        default=4,
    )
    parser.add_argument(
        "--vid",
        help="Indicator for saving the video of the diffusion process",
        action="store_true",
        dest="save_video",
    )
    parser.add_argument(
        "--export_assets",
        help="Indicator for saving raw assets of the prediction",
        action="store_true",
        dest="export_assets",
    )

    args = parser.parse_args()
    return args
