from pathlib import Path
import numpy as np
from numpy.core.shape_base import block
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage.measure import find_contours

import torch
from torchvision.transforms import functional as TF

from typing import Optional, Union
from PIL.Image import Image


def show_tensor_image(tensor: torch.Tensor, range_zero_one: bool = False):
    """Show a tensor of an image

    Args:
        tensor (torch.Tensor): Tensor of shape [N, 3, H, W] in range [-1, 1] or in range [0, 1]
    """
    if not range_zero_one:
        tensor = (tensor + 1) / 2
    tensor.clamp(0, 1)

    batch_size = tensor.shape[0]
    for i in range(batch_size):
        plt.title(f"Fig_{i}")
        pil_image = TF.to_pil_image(tensor[i])
        plt.imshow(pil_image)
        plt.show(block=True)


def show_edited_masked_image(
    refer_prompt: str,
    target_prompt: str,
    source_image: Image,
    edited_image: Image,
    mask: Optional[Image] = None,
    path: Optional[Union[str, Path]] = None,
    distance: Optional[str] = None,
):
    fig_idx = 1
    rows = 1
    cols = 3 if mask is not None else 2

    fig = plt.figure(figsize=(12, 5))
    # figure_title = f'Prompt: "{title}"'
    figure_title = f'Refer: "{refer_prompt}",    Target: "{target_prompt}"'
    if distance is not None:
        figure_title += f" ({distance})"
    plt.title(figure_title)
    plt.axis("off")

    fig.add_subplot(rows, cols, fig_idx)
    fig_idx += 1
    _set_image_plot_name("Source Image")
    plt.imshow(source_image)

    if mask is not None:
        fig.add_subplot(rows, cols, fig_idx)
        _set_image_plot_name("Mask")
        plt.imshow(mask)
        plt.gray()
        fig_idx += 1

    fig.add_subplot(rows, cols, fig_idx)
    _set_image_plot_name("Edited Image")
    plt.imshow(edited_image)
    
    if distance is not None:
        path_friendly_distance = distance.replace(".", "")
        ranked_path = path.with_name(path_friendly_distance + "_" + path.name)

    if path is not None:
        plt.savefig(ranked_path, bbox_inches="tight")
    else:
        plt.show(block=True)

    plt.close()


def _set_image_plot_name(name):
    plt.title(name)
    plt.xticks([])
    plt.yticks([])

    
    
    
    
# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(b.device)
    return b

def enlarge_bbox(out_bbox, scale=1.3, size=(256, 256)):
    x_c, y_c, w, h = out_bbox.unbind(1)
    w = w * scale
    h = h * scale
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    bbox = torch.stack(b, dim=1).clamp(0, 1)#.to(out_bbox.device)
    img_w, img_h = size
    bbox = bbox * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(bbox.device)
    return bbox

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def add_res(results, ax, color='green'):
    if True:
        bboxes = results['boxes']
        labels = results['labels']
        scores = results['scores']
    
    colors = ['purple', 'yellow', 'red', 'green', 'orange', 'pink']
    
    for i, (b, ll, ss) in enumerate(zip(bboxes, labels, scores)):
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color=colors[i], linewidth=3))
        cls_name = ll if isinstance(ll,str) else CLASSES[ll]
        text = f'{cls_name}: {ss:.2f}'
        # text = f'{ss:.2f}'
        print(text)
        ax.text(b[0], b[1], text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))
        
        
# def plot_results(pil_img, scores, boxes, labels, masks=None):
def plot_results(pil_img, scores, boxes, save_path, masks=None):
    # plt.figure()
    np_image = np.array(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if masks is None:
      masks = [None for _ in range(len(scores))]
    # assert len(scores) == len(boxes) == len(labels) == len(masks)
    # for s, (xmin, ymin, xmax, ymax), l, mask, c in zip(scores, boxes.tolist(), labels, masks, colors):
    if True:
        s = scores[0].cpu()
        xmin, ymin, xmax, ymax = boxes
        mask = masks[0].cpu()
        c = colors[0]
        # ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
        #                            fill=False, color=c, linewidth=3))
        # # text = f'{l}: {s:0.2f}'
        # text = f'{s:0.2f}'
        # ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

        # if mask is None:
        #   continue
        np_image = apply_mask(np_image, mask, c)

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
          # Subtract the padding and flip (y, x) to (x, y)
          verts = np.fliplr(verts) - 1
          p = Polygon(verts, facecolor="none", edgecolor=c)
          ax.add_patch(p)


    plt.imshow(np_image)
    plt.axis('off')
    # plt.show()
    plt.savefig(save_path)