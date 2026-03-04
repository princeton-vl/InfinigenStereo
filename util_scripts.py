# Some utilities to postprocess generated data.
# Authors: David Yan - Wrote utils and filtering code. 
# Lahav Lispon - Wrote the reprojection snippets.

from pathlib import Path

import numpy as np
from einops import einsum
from PIL import Image
from numpy.linalg import inv
import json

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def get_frame_path(scene_folder, cam_id, cam: int, frame_idx, data_type) -> Path:
    data_type_name, data_type_ext = data_type.split('_')
    imgname = f'{data_type_name}_{cam_id}_0_{frame_idx:04d}_{cam}.{data_type_ext}'
    return Path(scene_folder)/'frames'/data_type_name/f'camera_{cam}'/imgname

def transform(T, p):
    assert T.shape == (4,4)
    return einsum(p, T[:3,:3], 'H W j, i j -> H W i') + T[:3, 3]

def from_homog(x):
    return x[...,:-1] / x[...,[-1]]

def reproject(depth1, pose1, pose2, K1, K2):
    H, W = depth1.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    img_1_coords = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float64)
    cam1_coords = einsum(depth1, img_1_coords, inv(K1), 'H W, H W j, i j -> H W i')
    rel_pose = inv(pose2) @ pose1
    cam2_coords = transform(rel_pose, cam1_coords)
    return from_homog(einsum(cam2_coords, K2, 'H W j, i j -> H W i'))

def get_disp(seed, cam_id, frame):
    depth1_path = get_frame_path(seed, cam_id, 0, frame, 'Depth_npy')
    depth2_path = get_frame_path(seed, cam_id, 1, frame, 'Depth_npy')
    image1_path = get_frame_path(seed, cam_id, 0, frame, 'Image_png')
    image2_path = get_frame_path(seed, cam_id, 1, frame, 'Image_png')
    camview1_path = get_frame_path(seed, cam_id, 0, frame, 'camview_npz')
    camview2_path = get_frame_path(seed, cam_id, 1, frame, 'camview_npz')

    image1 = np.array(Image.open(image1_path)).astype(np.uint8)
    depth1 = np.load(depth1_path)
    depth2 = np.load(depth2_path)

    pose1 = np.load(camview1_path)['T']
    pose2 = np.load(camview2_path)['T']
    K1 = np.load(camview1_path)['K']
    K2 = np.load(camview2_path)['K']
    
    invalid = False

    # Check for invalid/corrupted depth
    if ((np.sum(np.isinf(depth1)) / depth1.size) * 100) > 95:
        invalid = True
    if ((np.sum(np.isnan(depth2)) / depth2.size) * 100) > 95:
        invalid = True

    # Simple heuristic for filtering dark scenes, more sophisticated ones can be used
    if np.mean(image1) < 15:
        invalid = True

    # Filter out cameras with objects too close
    if np.any(depth1 < 0.125):
        invalid = True
    if np.any(depth2 < 0.125):
        invalid = True

    if invalid:
        return None
    
    # Compute disparity through reprojection
    H, W, _ = image1.shape
    depth1 = cv2.resize(np.load(depth1_path), dsize=(W, H), interpolation=cv2.INTER_LINEAR)
    depth1 = np.nan_to_num(depth1, nan=1e4, posinf=1e4)
    img2_coords = reproject(depth1, pose1, pose2, K1, K2)
    height, width = img2_coords.shape[:2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    meshgrid_coords = np.stack((x, y), axis=-1)
    disparity = img2_coords - meshgrid_coords
    
    # Optionally, mask out sky 
    skymask = (depth1 < 1e3) 

    # Optionally, compute occlusion masks
    depth2 = cv2.resize(np.load(depth2_path), dsize=(W, H), interpolation=cv2.INTER_LINEAR)
    depth2 = np.nan_to_num(depth2, nan=1e4, posinf=1e4) 
    warped_depth2 = cv2.remap(
        depth2, img2_coords.astype(np.float32), None, interpolation=cv2.INTER_LINEAR
    )
    occ_mask = np.abs(warped_depth2 - depth1) < 0.1
    
    # Optionally, create mask based on object/material segmentation

    seg_path = get_frame_path(seed, cam_id, 0, frame, 'ObjectSegmentation_npy')
    json_path = get_frame_path(seed, cam_id, 0, frame, 'Objects_json')

    mat_json_path = get_frame_path(seed, cam_id, 0, frame, 'Materials_json')
    mat_seg_path = get_frame_path(seed, cam_id, 0, frame, 'MaterialSegmentation_npy')

    if mat_json_path.is_file() and mat_seg_path.is_file():
        segmentation = np.load(mat_seg_path)
        mats = json.load(open(mat_json_path))
        tags = [value['pass_index'] for key, value in mats.items() if 
                "metal" in key]
        material_mask = ~np.isin(segmentation, tags)

    if seg_path.is_file() and json_path.is_file():
        segmentation = np.load(seg_path)
        objects = json.load(open(json_path))
        tags = [value['object_index'] for key, value in objects.items() if 
            "exterior" in key]            
        
        object_mask = ~np.isin(segmentation, tags)


    assert disparity.shape == (*(depth1.shape), 2)

    # Need to invert disparity, since computed disparity is negative
    return -disparity[...,0], skymask, occ_mask, material_mask, object_mask


# Some more utilities to compute depth from disparity (can be used for post-hoc min-dist filtering of frames)

def camera_center(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return -R.T @ t                      

def baseline_vector(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    return camera_center(T2) - camera_center(T1)  

def baseline_length(T1: np.ndarray, T2: np.ndarray) -> float:
    return np.linalg.norm(baseline_vector(T1, T2))

def depth_from_disparity(disp_px: np.ndarray, f_px: float, B: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(disp_px > 0, f_px * B / disp_px, 0.0)
    return Z.astype(np.float32)

def get_disp(disp, seed, cam_id, frame):
    camview1_path = get_frame_path(seed, cam_id, 0, frame, 'camview_npz')
    camview2_path = get_frame_path(seed, cam_id, 1, frame, 'camview_npz')
    
    K_L = np.load(camview1_path)['K']
    K_R = np.load(camview2_path)['K']
    T_L = np.load(camview1_path)['T']
    T_R =  np.load(camview1_path)['T']

    B = baseline_length(T_L, T_R)         

    f_px = K_L[0, 0]             
    depth = depth_from_disparity(disp, f_px, B)

    reject = np.any(depth < 0.125) # e.g. reject frame based on min_dist

    return depth
