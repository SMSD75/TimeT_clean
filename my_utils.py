import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from torchvision.transforms import GaussianBlur
from typing import List
from IPython.display import display, display_markdown
import io
import os, sys
import requests
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import glob
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from typing import List
from torchvision.utils import draw_segmentation_masks
import cv2
from PIL import Image
import matplotlib
import numpy as np
import wandb
from torch import distributed as dist


def show_trainable_paramters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)



def process_attentions(attn_batch, spatial_res, threshold = 0.5, blur_sigma = 0.6):
    """
    Process [0,1] attentions to binary 0-1 mask. Applies a Guassian filter, keeps threshold % of mass and removes
    components smaller than 3 pixels.
    The code is adapted from https://github.com/facebookresearch/dino/blob/main/visualize_attention.py but removes the
    need for using ground-truth data to find the best performing head. Instead we simply average all head's attentions
    so that we can use the foreground mask during training time.
    :param attentions: torch 4D-Tensor containing the averaged attentions
    :param spatial_res: spatial resolution of the attention map
    :param threshold: the percentage of mass to keep as foreground.
    :param blur_sigma: standard deviation to be used for creating kernel to perform blurring.
    :return: the foreground mask obtained from the ViT's attention.
    """
    # Blur attentions
    # attns_processed = torch.cat(attns_group, dim = 0)
    attns_processed = sum(attn_batch[:, i] * 1 / attn_batch.size(1) for i in range(attn_batch.size(1)))
    attentions = attns_processed.reshape(-1, 1, spatial_res, spatial_res)
    attentions = GaussianBlur(7, sigma=(blur_sigma))(attentions)
    attentions = attentions.reshape(attentions.size(0), 1, spatial_res ** 2)
    # Keep threshold% of mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=-1, keepdim=True)
    cumval = torch.cumsum(val, dim=-1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    th_attn[:, 0] = torch.gather(th_attn[:, 0], dim=1, index=idx2[:, 0])
    th_attn = th_attn.reshape(attentions.size(0), 1, spatial_res, spatial_res).float()
    # Remove components with less than 3 pixels
    for j, th_att in enumerate(th_attn):
        labelled = label(th_att.cpu().numpy())
        for k in range(1, np.max(labelled) + 1):
            mask = labelled == k
            if np.sum(mask) <= 2:
                th_attn[j, 0][mask] = 0
    return th_attn.detach()



def preprocess(imgs):
    img_group = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = T.ToPILImage()(img.cpu())
        target_image_size = 224
        s = min(img.size)
        
        if s < target_image_size:
            raise ValueError(f'min dim for image {s} < {target_image_size}')
            
        r = target_image_size / s
        s = (round(r * img.size[1]), round(r * img.size[0]))
        img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
        img = TF.center_crop(img, output_size=2 * [target_image_size])
        img = torch.unsqueeze(T.ToTensor()(img), 0)
        img_group.append(map_pixels(img))
    return torch.cat(img_group, dim = 0)



def cosine_scheduler(base_value: float, final_value: float, max_iters: int):
    # Construct cosine schedule starting at base_value and ending at final_value with epochs * niter_per_ep values.
    iters = np.arange(max_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    return schedule


def denormalize_video(video):
    """
    video: [1, nf, c, h, w]
    """
    IMGNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    IMGNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    denormalized_video = video.cpu().detach() * IMGNET_STD + IMGNET_MEAN
    denormalized_video = (denormalized_video * 255).type(torch.uint8)
    denormalized_video = denormalized_video.squeeze(0)
    return denormalized_video

def overlay_video_cmap(cluster_maps, denormalized_video):
    """
    cluster_maps: [nf, h, w]
    denormalized_video: [nf, c, h, w]
    """
    ## generate 12 distinguishable colors
    colors = ["orange", "blue", "red", "yellow", "white", "green", "brown", "purple", "gold", "black", "pink", "cyan"]
        ## convert cluster_maps to [num_maps, h, w]
    masks = []
    cluster_ids = torch.unique(cluster_maps)
    for cluster_map in cluster_maps:
        mask = torch.zeros((cluster_ids.shape[0], cluster_map.shape[0], cluster_map.shape[1])) 
        mask = mask.type(torch.bool)
        for i, cluster_id in enumerate(cluster_ids):
                ## make a boolean mask for each cluster
                ## make a boolean mask for each cluster if cluster_map == cluster_id
            boolean_mask = (cluster_map == cluster_id)
            mask[i, :, :] = boolean_mask
        masks.append(mask)
    cluster_maps = torch.stack(masks)
            
    overlayed = [
                draw_segmentation_masks(img, masks=mask, alpha=0.5, colors=colors)
                for img, mask in zip(denormalized_video, cluster_maps)
            ]
    overlayed = torch.stack(overlayed)
    return cluster_maps,overlayed


def make_seg_maps(data, cluster_map, logging_directory, name, w_featmap=28, h_featmap=28):
    bs, fs, c, h, w = data.shape
    # cluster_map = torch.Tensor(cluster_map.reshape(bs, fs, w_featmap, h_featmap))
    # cluster_map = nn.functional.interpolate(cluster_map.type(torch.DoubleTensor), scale_factor=8, mode="nearest").detach().cpu()
    cluster_map = cluster_map
    for i, datum in enumerate(data):
        frame_buffer = []
        for j, frame in enumerate(datum):
            frame_buffer.append(localize_objects(frame.permute(1, 2, 0).detach().cpu(), cluster_map[i, j]))
        convert_list_to_video(frame_buffer, name + "_" + str(i), speed=1000/ datum.size(0), directory=logging_directory, wdb_log=False)
    

def visualize_sampled_videos(samples, path, name):
    # os.system(f'rm -r {path}')
    scale_255 = lambda x : (x * 255).astype('uint8')
    layer, height, width = samples[0].shape[-3:]
    if not os.path.isdir(path):
        os.mkdir(path)
    video = cv2.VideoWriter(path + name, 0, 1, (width,height))
    if len(samples.shape) == 4: ## sampling a batch of images and not clips
        frames = samples
    else: ## clip-wise sampling
        frames = samples[0][0]  

    for frame in frames:
        if len(frame.shape) == 3:
            frame_1 = frame.permute(1, 2, 0).numpy()
        else:
            frame_1 = frame[..., None].repeat(1, 1, 3).numpy()
        temp = scale_255(frame_1)
        # temp = frame_1
        video.write(temp)
    video.release()
    cv2.destroyAllWindows()

def localize_objects(input_img, cluster_map):

    colors = ["orange", "blue", "red", "yellow", "white", "green", "brown", "purple", "gold", "black"]
    ticks = np.unique(cluster_map.flatten()).tolist()

    dc = np.zeros(cluster_map.shape)
    for i in range(cluster_map.shape[0]):
        for j in range(cluster_map.shape[1]):
            dc[i, j] = ticks.index(cluster_map[i, j])

    colormap = matplotlib.colors.ListedColormap(colors)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 3))
    # plt.figure(figsize=(5,3))
    im = axes[0].imshow(dc, cmap=colormap, interpolation="none", vmin=-0.5, vmax=len(colors) - 0.5)
    cbar = fig.colorbar(im, ticks=range(len(colors)))
    axes[1].imshow(input_img)
    axes[2].imshow(dc, cmap=colormap, interpolation="none", vmin=-0.5, vmax=len(colors) - 0.5)
    axes[2].imshow(input_img, alpha=0.5)
    # plt.show(block=True)
    # plt.close()
    with io.BytesIO() as buffer:
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        return np.asarray(Image.open(buffer))


def convert_list_to_video(frames_list, name, speed, directory="", wdb_log=False):
    frames_list = [Image.fromarray(frame) for frame in frames_list]
    frames_list[0].save(f"{directory}{name}.gif", save_all=True, append_images=frames_list[1:], duration=speed, loop=0)
    if wdb_log:
        wandb.log({name: wandb.Video(f"{directory}{name}.gif", fps=4, format="gif")})


@torch.no_grad()
def sinkhorn(Q: torch.Tensor, nmb_iters: int, world_size=1) -> torch.Tensor:
    with torch.no_grad():
        Q = Q.detach().clone()
        sum_Q = torch.sum(Q)
        if world_size > 1:
            dist.all_reduce(sum_Q)
        Q /= sum_Q
        K, B = Q.shape
        u = torch.zeros(K).to(Q.device)
        r = torch.ones(K).to(Q.device) / K
        c = torch.ones(B).to(Q.device) / (B * world_size)

        if world_size > 1:
            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

        for _ in range(nmb_iters):
            if world_size > 1:
                u = curr_sum
            else:
                u = torch.sum(Q, dim=1)
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            if world_size > 1:
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)

        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


def find_optimal_assignment(scores, epsilon, sinkhorn_iterations, world_size=1):
    """
    Computes the Sinkhorn matrix Q.
    :param scores: similarity matrix
    :return: Sinkhorn matrix Q
    """
    with torch.no_grad():
        q = torch.exp(scores / epsilon).t()
        q = sinkhorn(q, sinkhorn_iterations, world_size=world_size)
        # q = torch.softmax(scores / epsilon, dim=0)
        # q = q / q.sum(dim=1, keepdim=True)
    return q
