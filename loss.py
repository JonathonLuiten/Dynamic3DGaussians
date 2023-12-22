import torch

from helpers import l1_loss_v1
from helpers import weighted_l2_loss_v1
from helpers import weighted_l2_loss_v2
from helpers import l1_loss_v1
from helpers import l1_loss_v2
from helpers import quat_mult
from helpers import params2rendervar

from external import calc_ssim
from external import build_rotation

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

L1_LOSS_WEIGHT:float =   0.8
SSIM_LOSS_WEIGHT:float = 0.2
LOSS_WEIGTHS = {
    'im': 1.0,
    'seg': 3.0,
    'rigid': 4.0,
    'rot': 4.0,
    'iso': 2.0,
    'floor': 2.0,
    'bg': 20.0,
    'soft_col_cons': 0.01
}

def apply_camera_parameters(image: torch.Tensor, params: dict, curr_data: dict) -> torch.Tensor:
    curr_id = curr_data['id']
    return torch.exp(params['cam_m'][curr_id])[:, None, None] * image + params['cam_c'][curr_id][:, None, None]

def compute_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    l1 = l1_loss_v1(rendered, target)
    ssim = 1.0 - calc_ssim(rendered, target)
    return L1_LOSS_WEIGHT * l1 + SSIM_LOSS_WEIGHT * ssim

def compute_rigid_loss(fg_pts, rot, variables):
    neighbor_pts = fg_pts[variables["neighbor_indices"]]
    curr_offset = neighbor_pts - fg_pts[:, None]
    curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
    return weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"], variables["neighbor_weight"])

def compute_rot_loss(rel_rot, variables):
    return weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None], variables["neighbor_weight"])

def compute_iso_loss(fg_pts, variables):
    neighbor_pts = fg_pts[variables["neighbor_indices"]]
    curr_offset = neighbor_pts - fg_pts[:, None]
    curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
    return weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])

def compute_floor_loss(fg_pts):
    return torch.clamp(fg_pts[:, 1], min=0).mean()

def compute_bg_loss(bg_pts, bg_rot, variables):
    return l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])


def get_loss(params:dict, curr_data:dict, variables:dict, is_initial_timestep:bool):

    losses = {}

    # Image
    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()
    image, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    image = apply_camera_parameters(image, params, curr_data)
    
    # Segmentation
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification
    segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)

    losses['im'] = compute_loss(image, curr_data['im'])
    losses['seg'] = compute_loss(seg, curr_data['seg'])
    
    if not is_initial_timestep:
        is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]
        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)

        losses['rigid'] = compute_rigid_loss(fg_pts, rot, variables)
        losses['rot'] = compute_rot_loss(rel_rot, variables)
        losses['iso'] = compute_iso_loss(fg_pts, variables)
        losses['floor'] = compute_floor_loss(fg_pts)
        
        bg_pts = rendervar['means3D'][~is_fg]
        bg_rot = rendervar['rotations'][~is_fg]
        losses['bg'] = compute_bg_loss(bg_pts, bg_rot, variables)
        losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])

    loss = sum([LOSS_WEIGTHS[k] * v for k, v in losses.items()])
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return loss, variables