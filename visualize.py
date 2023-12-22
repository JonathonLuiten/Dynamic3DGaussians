
import torch
import numpy as np
import open3d as o3d
from open3d.cuda.pybind.utility import Vector3dVector
import time
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, quat_mult
from external import build_rotation
from colormap import colormap
from copy import deepcopy

RENDER_MODE = 'color'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'depth'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'centers'  # 'color', 'depth' or 'centers'

ADDITIONAL_LINES = None  # None, 'trajectories' or 'rotations'
# ADDITIONAL_LINES = 'trajectories'  # None, 'trajectories' or 'rotations'
# ADDITIONAL_LINES = 'rotations'  # None, 'trajectories' or 'rotations'

REMOVE_BACKGROUND = False  # False or True
# REMOVE_BACKGROUND = True  # False or True

FORCE_LOOP = False  # False or True
# FORCE_LOOP = True  # False or True

width:int   = 640
height:int  = 360

near, far = 0.01, 100.0
view_scale = 3.9
fps = 20
traj_frac = 25  # 4% of points
traj_length = 15

# This tensor represents a set of homogeneous 2D coordinates for each pixel in the image.
# The last dimension (a constant value of 1) makes these coordinates homogeneous.
def_pix = torch.tensor(
            np.stack(
                np.meshgrid(np.arange(width) + 0.5,
                            np.arange(height) + 0.5,
                            1), -1
                    ).reshape(-1, 3)
            ).cuda().float()

pix_ones = torch.ones(height * width, 1).cuda().float()


def init_camera(
        y_angle:float=0.,       # degrees
        center_dist:float=2.4,  # meters?
        cam_height:float=1.3,   # meters?
        f_ratio:float=0.82
    ) -> tuple[np.ndarray, np.ndarray]:
    radians_y = y_angle * np.pi / 180. # radians
    
    world_2_cam = np.array([
                    [np.cos(radians_y),     0.,         -np.sin(radians_y),     0.],
                    [0.,                    1.,         0.,                     cam_height],
                    [np.sin(radians_y),     0.,         np.cos(radians_y),      center_dist],
                    [0.,                    0.,         0.,                     1.]
                    ])
    
    # Focal lengths
    fx = f_ratio * width  # Focal length in the x direction (in pixels)
    fy = f_ratio * width  # Assuming square pixels, so fx = fy. Change if different.

    # Optical center coordinates (typically the center of the image)
    cx = width / 2  # x-coordinate of the principal point (optical center)
    cy = height / 2 # y-coordinate of the principal point (optical center)

    camera_intrinsics = np.array([
            [fx, 0,  cx], 
            [0,  fy, cy], 
            [0,  0,  1]
        ])
    return world_2_cam, camera_intrinsics


def load_scene_data(
        seq:str,
        exp:str,
        seg_as_col:bool=False
        ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
    params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D': params['means3D'][t],
            'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        if REMOVE_BACKGROUND:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if REMOVE_BACKGROUND:
        is_fg = is_fg[is_fg]
    return scene_data, is_fg


def make_lineset(all_pts, cols, num_lines):
    linesets = []
    for pts in all_pts:
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def calculate_trajectories(scene_data, is_fg):
    in_pts = [data['means3D'][is_fg][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
    num_lines = len(in_pts[0])
    cols = np.repeat(colormap[np.arange(len(in_pts[0])) % len(colormap)][None], traj_length, 0).reshape(-1, 3)
    out_pts = []
    for t in range(len(in_pts))[traj_length:]:
        out_pts.append(np.array(in_pts[t - traj_length:t + 1]).reshape(-1, 3))
    return make_lineset(out_pts, cols, num_lines)


def calculate_rot_vec(scene_data, is_fg):
    in_pts = [data['means3D'][is_fg][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
    in_rotation = [data['rotations'][is_fg][::traj_frac] for data in scene_data]
    num_lines = len(in_pts[0])
    cols = colormap[np.arange(num_lines) % len(colormap)]
    inv_init_q = deepcopy(in_rotation[0])
    inv_init_q[:, 1:] = -1 * inv_init_q[:, 1:]
    inv_init_q = inv_init_q / (inv_init_q ** 2).sum(-1)[:, None]
    init_vec = np.array([-0.1, 0, 0])
    out_pts = []
    for t in range(len(in_pts)):
        cam_rel_qs = quat_mult(in_rotation[t], inv_init_q)
        rot = build_rotation(cam_rel_qs).cpu().numpy()
        vec = (rot @ init_vec[None, :, None]).squeeze()
        out_pts.append(np.concatenate((in_pts[t] + vec, in_pts[t]), 0))
    return make_lineset(out_pts, cols, num_lines)


def render(
        world_2_cam:np.ndarray,
        instrinsics:np.ndarray,
        timestep_data:dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        cam = setup_camera(width, height, instrinsics, world_2_cam, near, far)
        image, _, depth, = Renderer(raster_settings=cam)(**timestep_data)
        # [3, height, width], [1, height, width]
        return image, depth


def rgbd_2_pointcloud(
        image:torch.Tensor,
        depth:torch.Tensor,
        world_2_cam:np.ndarray,
        instrinsics:np.ndarray,
        show_depth:bool = False,
        project_to_cam_w_scale:float | None = None
    ) -> tuple[Vector3dVector, Vector3dVector]:
    depth_near:float    = 1.5
    depth_far:float     = 6

    # intrinsic matrix is useful when you want to back-project 2D points from the image plane to 3D rays in camera space
    inv_instrinsics = torch.inverse(torch.tensor(instrinsics).cuda().float())
    # this matrix transforms points from the camera frame back to the world frame. 
    inv_world_2_cam = torch.inverse(torch.tensor(world_2_cam).cuda().float())
    
    radial_depth = depth[0].reshape(-1)
    # This operation back-projects 'def_pix' 2D coordinates into 3D camera coordinates,
    # effectively producing rays (direction vectors) for each pixel that emanate from the camera's optical center
    def_rays = (inv_instrinsics @ def_pix.T).T
    def_radial_rays = def_rays / torch.linalg.norm(def_rays, ord=2, dim=-1)[:, None]
    
    # Computes the actual 3D coordinates of the points in the scene in the camera's coordinate system using the depth information.
    camera_space_points = def_radial_rays * radial_depth[:, None]
    depth_values_along_optical_axis = camera_space_points[:, 2]
    
    if project_to_cam_w_scale is not None:
        camera_space_points = project_to_cam_w_scale * camera_space_points / depth_values_along_optical_axis[:, None]
    
    # Converts the 3D coordinates in camera_space_points into 4D homogeneous coordinates.
    camera_space_points_homogeneous = torch.concat((camera_space_points, pix_ones), 1)
    # from the camera's coordinate space to the world coordinate space.
    world_space_points = (inv_world_2_cam @ camera_space_points_homogeneous.T).T[:, :3]
    
    if show_depth: # based on their depth values.
        colors = ((depth_values_along_optical_axis - depth_near) / (depth_far - depth_near))[:, None].repeat(1, 3)
    else:
        colors = torch.permute(image, (1, 2, 0)).reshape(-1, 3)
    
    world_space_points: Vector3dVector =    o3d.utility.Vector3dVector(world_space_points.contiguous().double().cpu().numpy())
    colors: Vector3dVector =                o3d.utility.Vector3dVector(colors.contiguous().double().cpu().numpy())
    
    return world_space_points, colors


def visualize(seq:str, exp:str):
    scene_data, is_fg = load_scene_data(seq, exp)
    vis = o3d.visualization.Visualizer() #type: ignore
    vis.create_window(width=int(width * view_scale), height=int(height * view_scale), visible=True)

    world_2_cam, intrinsics = init_camera(y_angle=60.)
    image, depth = render(world_2_cam, intrinsics, scene_data[0])
    init_points, init_colors = rgbd_2_pointcloud(image, depth, world_2_cam, intrinsics, show_depth=(RENDER_MODE == 'depth'))

    # collection of points in 3D space, each potentially having associated properties like color, normals, etc
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = init_points
    point_cloud.colors = init_colors
    vis.add_geometry(point_cloud)

    lines: o3d.geometry.LineSet | None = None
    linesets:list[o3d.geometry.LineSet] | None = None
    if ADDITIONAL_LINES is not None:
        if ADDITIONAL_LINES == 'trajectories':
            linesets = calculate_trajectories(scene_data, is_fg)
        elif ADDITIONAL_LINES == 'rotations':
            linesets = calculate_rot_vec(scene_data, is_fg)
        else:
            raise ValueError(f"Unsupported value for ADDITIONAL_LINES")
        lines = o3d.geometry.LineSet()
        lines.points = linesets[0].points   #type: ignore
        lines.colors = linesets[0].colors   #type: ignore
        lines.lines = linesets[0].lines     #type: ignore
        vis.add_geometry(lines)

    # adjust the focal length and optical center according to the view_scale.
    view_intrinsics = intrinsics * view_scale
    view_intrinsics[2, 2] = 1 # don't scale depth
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    cparams.extrinsic = world_2_cam
    cparams.intrinsic.intrinsic_matrix = view_intrinsics
    cparams.intrinsic.height = int(height * view_scale)
    cparams.intrinsic.width = int(width * view_scale)
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = view_scale
    render_options.light_on = False

    start_time = time.time()
    num_timesteps = len(scene_data)
    while True:
        passed_time = time.time() - start_time
        passed_frames = passed_time * fps
        if ADDITIONAL_LINES == 'trajectories':
            time_step = int(passed_frames % (num_timesteps - traj_length)) + traj_length  # Skip t that don't have full traj.
        else:
            time_step = int(passed_frames % num_timesteps)

        if FORCE_LOOP:
            num_loops = 1.4
            y_angle = 360 * time_step * num_loops / num_timesteps
            world_2_cam, instrinsics = init_camera(y_angle)
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            cam_params.extrinsic = world_2_cam
            view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        else:  # Interactive control
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            view_instrinsics = cam_params.intrinsic.intrinsic_matrix
            instrinsics = view_instrinsics / view_scale
            instrinsics[2, 2] = 1
            world_2_cam = cam_params.extrinsic

        if RENDER_MODE == 'centers':
            points = o3d.utility.Vector3dVector(scene_data[time_step]['means3D'].contiguous().double().cpu().numpy())
            colors = o3d.utility.Vector3dVector(scene_data[time_step]['colors_precomp'].contiguous().double().cpu().numpy())
        else:
            image, depth = render(world_2_cam, instrinsics, scene_data[time_step])
            points, colors = rgbd_2_pointcloud(image, depth, world_2_cam, instrinsics, show_depth=(RENDER_MODE == 'depth'))
        
        point_cloud.points = points
        point_cloud.colors = colors
        vis.update_geometry(point_cloud)

        if ADDITIONAL_LINES is not None and \
                      lines is not None and \
                   linesets is not None:
            if ADDITIONAL_LINES == 'trajectories':
                lt = time_step - traj_length
            else:
                lt = time_step
            lines.points = linesets[lt].points
            lines.colors = linesets[lt].colors
            lines.lines = linesets[lt].lines
            vis.update_geometry(lines)

        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()
    del view_control
    del vis
    del render_options


if __name__ == "__main__":
    exp_name = "pretrained"
    for sequence in ["basketball", "boxes", "football", "juggle", "softball", "tennis"]:
        visualize(sequence, exp_name)
