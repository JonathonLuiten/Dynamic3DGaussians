import os
import json
import random
import copy

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from helpers import setup_camera
from helpers import o3d_knn
from helpers import params2rendervar
from helpers import params2cpu
from helpers import save_params

from external import calc_psnr
from external import densify
from external import update_params_and_optimizer

from loss import get_loss

MAX_CAMS:int =          50
NUM_NEAREST_NEIGH:int = 3
SCENE_SIZE_MULT:float = 1.1

# Camera
NEAR:float =    1.0
FAR:float =     100.

# Training Hyperparams
INITIAL_TIMESTEP_ITERATIONS =   10_000
TIMESTEP_ITERATIONS =           2_000


def construct_timestep_dataset(timestep:int, metadata:dict, sequence:str) -> list[dict]:
    dataset_entries = []
    for camera_id in range(len(metadata['fn'][timestep])):
        width, height, intrinsics, extrinsics = metadata['w'], metadata['h'], metadata['k'][timestep][camera_id], metadata['w2c'][timestep][camera_id]
        camera = setup_camera(width, height, intrinsics, extrinsics, near=NEAR, far=FAR)
        
        filename = metadata['fn'][timestep][camera_id]
        
        image = np.array(copy.deepcopy(Image.open(f"./data/{sequence}/ims/{filename}")))
        image_tensor = torch.tensor(image).float().cuda().permute(2, 0, 1) / 255.
        
        segmentation = np.array(copy.deepcopy(Image.open(f"./data/{sequence}/seg/{filename.replace('.jpg', '.png')}"))).astype(np.float32)
        segmentation_tensor = torch.tensor(segmentation).float().cuda()
        segmentation_color = torch.stack((segmentation_tensor, torch.zeros_like(segmentation_tensor), 1 - segmentation_tensor))
        
        dataset_entries.append({'cam': camera, 'im': image_tensor, 'seg': segmentation_color, 'id': camera_id})
    
    return dataset_entries


def initialize_batch_sampler(dataset:list[dict]) -> list[int]:
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return indices


def get_data_point(batch_sampler:list[int], dataset:list[dict]) -> dict:
    if len(batch_sampler) < 1: batch_sampler = initialize_batch_sampler(dataset)
    return dataset[batch_sampler.pop()]


def initialize_params(sequence:str, metadata:dict) -> tuple[dict, dict]:
    init_pt_cld:np.ndarray = np.load(f"./data/{sequence}/init_pt_cld.npz")["data"]
    segmentation = init_pt_cld[:, 6]
    square_distance, _ = o3d_knn(init_pt_cld[:, :3], NUM_NEAREST_NEIGH)
    mean_square_distance = square_distance.mean(-1).clip(min=1e-7)
    params = {
        'means3D': init_pt_cld[:, :3],
        'rgb_colors': init_pt_cld[:, 3:6],
        'seg_colors': np.stack((segmentation, np.zeros_like(segmentation), 1 - segmentation), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (segmentation.shape[0], 1)),
        'logit_opacities': np.zeros((segmentation.shape[0], 1)),
        'log_scales': np.tile(np.log(np.sqrt(mean_square_distance))[..., None], (1, 3)),
        'cam_m': np.zeros((MAX_CAMS, 3)),
        'cam_c': np.zeros((MAX_CAMS, 3)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    cam_centers = np.linalg.inv(metadata['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = SCENE_SIZE_MULT * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    return params, variables


def initialize_optimizer(params:dict, variables:dict):
    lrs = {
        'means3D': 0.00016 * variables['scene_radius'],
        'rgb_colors': 0.0025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)



def initialize_per_timestep(
    params: dict[str, torch.Tensor], 
    variables: dict[str, torch.Tensor], 
    optimizer: torch.optim.Optimizer
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

    current_points = params['means3D']
    current_rotations_normalized = torch.nn.functional.normalize(params['unnorm_rotations'])

    # Calculate momentum-like updates
    new_points = current_points + (current_points - variables["prev_pts"])
    new_rotations = torch.nn.functional.normalize(current_rotations_normalized + (current_rotations_normalized - variables["prev_rot"]))

    # Extract foreground entities' info
    foreground_mask = params['seg_colors'][:, 0] > 0.5
    previous_inverse_rotations_foreground = current_rotations_normalized[foreground_mask]
    previous_inverse_rotations_foreground[:, 1:] = -1 * previous_inverse_rotations_foreground[:, 1:]
    foreground_points = current_points[foreground_mask]
    previous_offsets = foreground_points[variables["neighbor_indices"]] - foreground_points[:, None]

    # Update previous values in the variables dictionary
    variables['prev_inv_rot_fg'] = previous_inverse_rotations_foreground.detach()
    variables['prev_offset'] = previous_offsets.detach()
    variables["prev_col"] = params['rgb_colors'].detach()
    variables["prev_pts"] = current_points.detach()
    variables["prev_rot"] = current_rotations_normalized.detach()

    # Update the params dictionary
    updated_params = {'means3D': new_points, 'unnorm_rotations': new_rotations}
    params = update_params_and_optimizer(updated_params, params, optimizer)

    return params, variables



def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means3D'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['unnorm_rotations']).detach()
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables


def report_progress(params, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)


def train(sequence:str, exp_name:str):
    if os.path.exists(f"./output/{exp_name}/{sequence}"):
        print(f"Experiment '{exp_name}' for sequence '{sequence}' already exists. Exiting.")
        return
   
    metadata = json.load(open(f"./data/{sequence}/train_meta.json", 'r'))
    num_timesteps = len(metadata['fn'])

    params, variables = initialize_params(sequence, metadata)
    optimizer = initialize_optimizer(params, variables)
    output_params = []
    
    for timestep in range(num_timesteps):
        dataset = construct_timestep_dataset(timestep, metadata, sequence)
        batch_sampler = initialize_batch_sampler(dataset)
        is_initial_timestep = (timestep == 0)
        if not is_initial_timestep:
            # "momentum-based update"
            params, variables = initialize_per_timestep(params, variables, optimizer)
        
        num_iter_per_timestep = INITIAL_TIMESTEP_ITERATIONS if is_initial_timestep else TIMESTEP_ITERATIONS
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {timestep}")
        
        for i in range(num_iter_per_timestep):
            curr_data = get_data_point(batch_sampler, dataset)
            loss, variables = get_loss(params, curr_data, variables, is_initial_timestep)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar)
                if is_initial_timestep:
                    params, variables = densify(params, variables, optimizer, i)
        
        progress_bar.close()
        output_params.append(params2cpu(params, is_initial_timestep))
        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)
            
    save_params(output_params, sequence, exp_name)


def main():
    exp_name = "exp1"
    for sequence in ["basketball", "boxes", "football", "juggle", "softball", "tennis"]:
        train(sequence, exp_name)
        torch.cuda.empty_cache()
    
    
if __name__ == "__main__":
    main()