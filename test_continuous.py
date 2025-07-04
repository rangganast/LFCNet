import os
import time
import yaml
import mathutils
import numpy as np
from numpy.linalg import inv
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms as T
from torchvision.utils import save_image

from test_modules.quaternion_distances import quaternion_distance

from test_modules import so3

# from dataset.dataset_test import KITTI360_Fisheye_Dataset
from dataset.dataset import KITTI360_Fisheye_Dataset
from models.model import Net  
 
def load_calib_gt(rootdir):
    # camera extrinsic and intrinsic parameter transformation
    cam_to_pose_path = os.path.join(rootdir, "calibration", "calib_cam_to_pose.txt")
    cam_to_velo_path = os.path.join(rootdir, "calibration", "calib_cam_to_velo.txt")

    data_cam_to_pose = {}
    data_cam_to_velo = {}

    with open(cam_to_pose_path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':',1)

            try:
                data_cam_to_pose[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    with open(cam_to_velo_path, 'r') as f:
        for line in f.readlines():
            try:
                data_cam_to_velo = np.array([float(x) for x in line.split()])
            except ValueError:
                pass

    T_cam0_to_pose = np.reshape(data_cam_to_pose['image_00'], (3, 4))
    T_cam1_to_pose = np.reshape(data_cam_to_pose['image_01'], (3, 4))
    T_cam2_to_pose = np.reshape(data_cam_to_pose['image_02'], (3, 4))
    T_cam3_to_pose = np.reshape(data_cam_to_pose['image_03'], (3, 4))
    T_cam0_to_velo = np.reshape(data_cam_to_velo, (3, 4))

    last_row = np.array([0, 0, 0, 1])

    T_cam0_to_pose = np.vstack((T_cam0_to_pose, last_row))
    T_cam1_to_pose = np.vstack((T_cam1_to_pose, last_row))
    T_cam2_to_pose = np.vstack((T_cam2_to_pose, last_row))
    T_cam3_to_pose = np.vstack((T_cam3_to_pose, last_row))
    T_cam0_to_velo = np.vstack((T_cam0_to_velo, last_row))

    T_velo_to_cam0 = inv(T_cam0_to_velo)
    T_velo_to_cam1 = inv(T_cam1_to_pose).dot(T_cam0_to_pose).dot(inv(T_cam0_to_velo))
    T_velo_to_cam2 = inv(T_cam2_to_pose).dot(T_cam0_to_pose).dot(inv(T_cam0_to_velo))
    T_velo_to_cam3 = inv(T_cam3_to_pose).dot(T_cam0_to_pose).dot(inv(T_cam0_to_velo))

    return T_velo_to_cam0, T_velo_to_cam1, T_velo_to_cam2, T_velo_to_cam3


def generate_misalignment(max_rot = 30, max_trans = 0.5):
    rot_z = np.random.uniform(-max_rot, max_rot) * (3.141592 / 180.0)
    rot_y = np.random.uniform(-max_rot, max_rot) * (3.141592 / 180.0)
    rot_x = np.random.uniform(-max_rot, max_rot) * (3.141592 / 180.0)
    trans_x = np.random.uniform(-max_trans, max_trans)
    trans_y = np.random.uniform(-max_trans, max_trans)
    trans_z = np.random.uniform(-max_trans, max_trans)

    R_perturb = mathutils.Euler((rot_x, rot_y, rot_z)).to_matrix()
    t_perturb = np.array([trans_x, trans_y, trans_z])
        
    return torch.tensor(R_perturb).type(torch.float32), torch.tensor(t_perturb).type(torch.float32)

def load_fisheye_intrinsic(rootdir, camera_id):
    intrinsic_path = os.path.join(rootdir, 
                                    "calibration", 
                                    "image_%s.yaml"%(camera_id))
    
    with open(intrinsic_path, 'r') as file:
        intrinsic = yaml.safe_load(file)

    # 2D image dimension
    H = intrinsic['image_height']
    W = intrinsic['image_width']
    img_size = (H, W)
    
    # mirror parameters
    xi = intrinsic['mirror_parameters']['xi']

    # distortion parameters
    k1 = intrinsic['distortion_parameters']['k1']
    k2 = intrinsic['distortion_parameters']['k2']
    p1 = intrinsic['distortion_parameters']['p1']
    p2 = intrinsic['distortion_parameters']['p2']

    # projection parameters
    gamma1 = intrinsic['projection_parameters']['gamma1']
    gamma2 = intrinsic['projection_parameters']['gamma2']
    u0 = intrinsic['projection_parameters']['u0']
    v0 = intrinsic['projection_parameters']['v0']
    
    # intrinsic matrix (K) formulation
    K_int = np.array([[gamma1, 0.0, u0],[0.0, gamma2, v0], [0.0, 0.0, 1.0]])

    # mirror & distortion parameters dictionary
    distort_params = {'xi': xi, 'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2}

    return img_size, K_int, distort_params

class depth_rgb_proj:
    def __init__(self, image_size, K_int, distort_params):
        self.H, self.W = image_size
        
        # intrinsic parameters
        self.K_int = K_int 
        self.K_int = torch.tensor(self.K_int).type(torch.float32)

        # mirror and distortion parameters
        self.xi = distort_params['xi']
        self.k1 = distort_params['k1']
        self.k2 = distort_params['k2']
        self.p1 = distort_params['p1']
        self.p2 = distort_params['p2']
        
    def __call__(self, point_cloud, rgb_img, device, image_path='test_calib.png'):
        rgb_img = rgb_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    
        point_cloud = point_cloud.to(device)
        # T_ext = T_ext.squeeze(0).to(device)
        self.K_int = self.K_int.to(device)
     
        # extrinsic transformation from velodyne reference frame to fisheye camera reference frame
        # pcd_fisheye = torch.matmul(T_ext, point_cloud.T).T  # (P_velo -> P_fisheye)
        pcd_fisheye = point_cloud[:,:3]
        z_axis = pcd_fisheye[:,2]  # Extract the z_axis to determine 3D points that located in front of the camera.

        # Project to unit sphere (every vector magnitude is 1)
        pcd_sphere = pcd_fisheye / torch.linalg.norm(pcd_fisheye, dim=1, keepdim=True)

        # Reference frame center moved to Cp = xi => (X, Y, Z) --> (X, Y, Z + Xi)

        pcd_sphere[:,2] += self.xi

        # Project into normalized plane => (X/(Z + Xi), Y/(Z + Xi), 1)
        norm_plane = pcd_sphere / pcd_sphere[:, 2].unsqueeze(1)

        # distortion ((u',v') = radial(u,v) + tangential(u,v))
        distorted_plane = self.distortion(norm_plane)

        # final projection using generalized camera projection (intrinsic matrix K)
        distorted_pixels = torch.cat((distorted_plane, norm_plane[:, 2].unsqueeze(1)), dim=1)
        pixel_proj = torch.matmul(self.K_int, distorted_pixels.T).T

        # Convert pixel_proj to integer indices for projection
        u = pixel_proj[:, 0].to(torch.int32)
        v = pixel_proj[:, 1].to(torch.int32)

        # Compute depth for each point in the point cloud
        depth = torch.norm(point_cloud, dim=1)

        condition = (0<=u)*(u<self.W)*(0<=v)*(v<self.H)*(depth>0)*(z_axis>=0)
        # print(np.min(z_axis))

        u_proj = u[condition]
        v_proj = v[condition]
        d_proj = depth[condition]
        
        max_depth = torch.max(d_proj).to(device)
        d_proj = torch.clamp((d_proj / max_depth) * 255, 0, 255).to(device)

        # # image array generation
        # image_tensor = torch.zeros(self.H, self.W, dtype=torch.float32).to(device)
        # image_tensor[v_proj,u_proj] = d_proj

        # # expand dimension to (1400, 1400, 3)
        # image_tensor = torch.unsqueeze(image_tensor, 0)
        # image_tensor = image_tensor.expand(3, -1, -1)
        
        # save_image(image_tensor, "test_image.png")
        
        u_proj = u_proj.detach().cpu().numpy()
        v_proj = v_proj.detach().cpu().numpy()
        d_proj = d_proj.detach().cpu().numpy()
        
        fig = plt.figure(figsize=(14,14),dpi=100,frameon=False)
        # fig.set_size(self.W, self.H)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(rgb_img)
        ax.scatter([u_proj],[v_proj],c=[d_proj],cmap='rainbow_r',alpha=0.3,s=10)
        # fig.show()
        fig.savefig(image_path)
        
    def distortion(self, norm_plane):
        # Compute r = sqrt(x^2 + y^2) for each pixel
        r = torch.sqrt(norm_plane[:, 0]**2 + norm_plane[:, 1]**2)

        # Radial distortion
        D = 1 + self.k1 * (r**2) + self.k2 * (r**4)

        # Tangential distortion
        dx = 2 * self.p1 * norm_plane[:, 0] * norm_plane[:, 1] + self.p2 * ((r**2) + 2 * (norm_plane[:, 0]**2))
        dy = self.p1 * ((r**2) + 2 * (norm_plane[:, 1]**2)) + 2 * self.p2 * norm_plane[:, 0] * norm_plane[:, 1]

        # Total distortion
        x_distorted = D * norm_plane[:, 0] + dx
        y_distorted = D * norm_plane[:, 1] + dy

        # Combine into a distorted plane tensor
        distorted_plane = torch.stack([x_distorted, y_distorted], dim=1)
        return distorted_plane
        
class depth_image_projection_fisheye_torch:
    def __init__(self, image_size, K_int, distort_params):
        self.H, self.W = image_size
        
        # intrinsic parameters
        self.K_int = K_int 

        # mirror and distortion parameters
        self.xi = distort_params['xi']
        self.k1 = distort_params['k1']
        self.k2 = distort_params['k2']
        self.p1 = distort_params['p1']
        self.p2 = distort_params['p2']
        
    def __call__(self, point_cloud, T_ext, device):
        point_cloud = point_cloud.to(device)
        T_ext = T_ext.to(device)
        self.K_int = self.K_int.to(device)
     
        # extrinsic transformation from velodyne reference frame to fisheye camera reference frame
        n_points = point_cloud.shape[0]
        pcd_fisheye = torch.matmul(T_ext, torch.hstack((point_cloud, torch.ones((n_points, 1)).to(device))).T).T  # (P_velo -> P_fisheye)
        pcd_fisheye = pcd_fisheye[:,:3]
        z_axis = pcd_fisheye[:,2]  # Extract the z_axis to determine 3D points that located in front of the camera.

        # Project to unit sphere (every vector magnitude is 1)
        pcd_sphere = pcd_fisheye / torch.linalg.norm(pcd_fisheye, dim=1, keepdim=True)

        # Reference frame center moved to Cp = xi => (X, Y, Z) --> (X, Y, Z + Xi)
        pcd_sphere[:,2] += self.xi

        # Project into normalized plane => (X/(Z + Xi), Y/(Z + Xi), 1)
        norm_plane = pcd_sphere / pcd_sphere[:, 2].unsqueeze(1)

        # distortion ((u',v') = radial(u,v) + tangential(u,v))
        distorted_plane = self.distortion(norm_plane)

        # final projection using generalized camera projection (intrinsic matrix K)
        distorted_pixels = torch.cat((distorted_plane, norm_plane[:, 2].unsqueeze(1)), dim=1)
        pixel_proj = torch.matmul(self.K_int, distorted_pixels.T).T

        # Convert pixel_proj to integer indices for projection
        u = pixel_proj[:, 0].to(torch.int32)
        v = pixel_proj[:, 1].to(torch.int32)

        # Compute depth for each point in the point cloud
        depth = torch.norm(point_cloud, dim=1)

        condition = (0<=u)*(u<self.W)*(0<=v)*(v<self.H)*(depth>0)*(z_axis>=0)
        # print(np.min(z_axis))

        u_proj = u[condition]
        v_proj = v[condition]
        d_proj = depth[condition]

        max_depth = torch.max(d_proj).to(device)
        d_proj = torch.clamp((d_proj / max_depth) * 255, 0, 255).to(device)

        # image array generation
        image_tensor = torch.zeros(self.H, self.W, dtype=torch.float32).to(device)
        image_tensor[v_proj,u_proj] = d_proj

        # expand dimension to (1400, 1400, 3)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        image_tensor = image_tensor.expand(3, -1, -1)

        return image_tensor

    def distortion(self, norm_plane):
        # Compute r = sqrt(x^2 + y^2) for each pixel
        r = torch.sqrt(norm_plane[:, 0]**2 + norm_plane[:, 1]**2)

        # Radial distortion
        D = 1 + self.k1 * (r**2) + self.k2 * (r**4)

        # Tangential distortion
        dx = 2 * self.p1 * norm_plane[:, 0] * norm_plane[:, 1] + self.p2 * ((r**2) + 2 * (norm_plane[:, 0]**2))
        dy = self.p1 * ((r**2) + 2 * (norm_plane[:, 1]**2)) + 2 * self.p2 * norm_plane[:, 0] * norm_plane[:, 1]

        # Total distortion
        x_distorted = D * norm_plane[:, 0] + dx
        y_distorted = D * norm_plane[:, 1] + dy

        # Combine into a distorted plane tensor
        distorted_plane = torch.stack([x_distorted, y_distorted], dim=1)
        return distorted_plane

class pcd_extrinsic_transform_realign: # transform PCD into fisheye camera reference frame.
    def __init__(self, crop = True):
        self.crop = crop

    def __call__(self, point_cloud, T_ext):
        # Apply extrinsic transformation
        pcd_cam = torch.matmul(T_ext.inverse(), point_cloud.T).T  # (N, 4)
        pcd_cam = pcd_cam[:, :3]  # Extract 3D coordinates (N, 3)

        # Extract the z-axis to determine points in front of the camera
        z_axis = pcd_cam[:, 2]

        # Apply cropping if enabled
        if self.crop:
            condition = z_axis >= 0
            new_pcd = pcd_cam[condition]
        else:
            new_pcd = pcd_cam

        return new_pcd

def load_checkpoint(checkpoint_path, model):
    print("=> Loading checkpoint")
    print()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    last_epoch = checkpoint["epoch"]
    last_epoch_loss = checkpoint["loss"]

    return model, last_epoch, last_epoch_loss

def test_metrics(T_pred, T_gt):
    R_pred = T_pred[:,:3,:3]  # (B,3,3) rotation
    t_pred = T_pred[:,:3,3]   # (B,3) translation
    R_gt = T_gt[:,:3,:3]  # (B,3,3) rotation
    t_gt = T_gt[:,:3,3]   # (B,3) translation

    # Euclidian Distance / Absolute Distance Error Rate
    t_error = F.l1_loss(t_pred, t_gt, reduction='none')
    # t_error = torch.sqrt(t_error)
    e_x = t_error[:,0].mean(dim=0)
    e_y = t_error[:,1].mean(dim=0)
    e_z = t_error[:,2].mean(dim=0)
    # e_t = t_error.mean(dim=1).mean(dim=0)
    e_t = torch.norm(t_pred - t_gt, p=2)
    
    q_pred = rot2qua_torch(R_pred.squeeze())
    q_gt = rot2qua_torch(R_gt.squeeze())

    # Euler Angles Error Rate
    
    q_pred = q_pred.unsqueeze(0)
    q_gt = q_gt.unsqueeze(0)
    
    # e_r = quaternion_distance(q_gt, q_pred, DEVICE)
    RIR = torch.bmm(torch.inverse(R_pred), R_gt)

    yaws = torch.atan2(RIR[:,1,0], RIR[:,0,0])
    pitches = torch.atan2(-RIR[:,2,0], torch.sqrt(RIR[:,2,0]*RIR[:,2,0] + RIR[:,2,2]*RIR[:,2,2]))
    rolls = torch.atan2(RIR[:,2,1], RIR[:,2,2])

    e_yaw = (torch.abs(yaws)).mean(dim=0)
    e_pitch = (torch.abs(pitches)).mean(dim=0)
    e_roll = (torch.abs(rolls)).mean(dim=0)
    e_r = (e_yaw + e_roll + e_pitch)/3.0

    # Geodesic Error Rate
    RTR = torch.bmm(torch.transpose(R_pred, 1, 2), R_gt)
    dR = so3.log(RTR)
    dR = F.mse_loss(dR,torch.zeros_like(dR).to(dR),reduction='none').mean(dim=1)  # (B,3) -> (B,1)
    dR = torch.sqrt(dR).mean(dim=0)

    return e_x, e_y, e_z, e_t, e_yaw, e_pitch, e_roll, e_r, dR

def qua2rot_torch(q):
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [3x3] homogeneous rotation matrix
    """
    assert q.shape == torch.Size([4]), "Not a valid quaternion"
    if q.norm() != 1.:
        q = q / q.norm()
    mat = torch.zeros((3, 3), device=q.device)
    mat[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    mat[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    mat[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    mat[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    mat[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    mat[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    mat[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    mat[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    mat[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    return mat

def rot2qua_torch(matrix):
    """
    Convert a rotation matrix to quaternion.
    Args:
        matrix (torch.Tensor): [4x4] transformation matrix or [3,3] rotation matrix.

    Returns:
        torch.Tensor: shape [4], normalized quaternion
    """
    if matrix.shape == (4, 4):
        R = matrix[:-1, :-1]
    elif matrix.shape == (3, 3):
        R = matrix
    else:
        raise TypeError("Not a valid rotation matrix")
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    q = torch.zeros(4, device=matrix.device)
    if tr > 0.:
        S = (tr+1.0).sqrt() * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]).sqrt() * 2
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]).sqrt() * 2
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]).sqrt() * 2
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q / q.norm()

# def rotate_points_torch()

def test(weights, loader, batch_size):
    num_stages = len(weights)
    # Pre-load all models for efficiency
    models = []
    for weight in weights:
        model = Net().to(DEVICE)
        model, _, _ = load_checkpoint(weight, model)
        model.eval()
        models.append(model)

    # Metrics storage
    ex_epoch = []
    ey_epoch = []
    ez_epoch = []
    et_epoch = []
    eyaw_epoch = []
    eroll_epoch = []
    epitch_epoch = []
    er_epoch = []

    img_size, K, distort_params = load_fisheye_intrinsic(rootdir="/home/rangganast/rangganast/dataset/KITTI-360", camera_id="02")
    K = torch.from_numpy(K).type(torch.float32)
    gen_depth_img = depth_image_projection_fisheye_torch(image_size=img_size, K_int=K, distort_params=distort_params)
    T_velo_to_cam_gt = load_calib_gt(rootdir="/home/rangganast/rangganast/dataset/KITTI-360")[2]
    T_velo_to_cam_gt = torch.tensor(T_velo_to_cam_gt).type(torch.float32)
    depth_transform = T.Compose([T.Resize((350, 350))])

    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(loader, desc="Multi Frame")):
            # Prepare data
            pcd = batch_data["pcd"].squeeze(0).to(DEVICE)
            rgb_img = batch_data["img"].to(DEVICE)
            # Start with the largest misalignment and corresponding depth image
            
            if i == 0:
                depth_img = batch_data["depth_img_error"].to(DEVICE)
                T_mis = batch_data["T_mis"].squeeze(0).to(DEVICE)
            else:
                depth_img = gen_depth_img(pcd, T_mis, DEVICE)
                depth_img = depth_transform(depth_img)
                depth_img = depth_img.unsqueeze(0).to(DEVICE)

            # Iterative refinement
            for stage in range(num_stages):
                model = models[stage]
                # Predict delta transformation
                delta_q_pred, delta_t_pred = model(rgb_img, depth_img)
                delta_q_pred = delta_q_pred[0].detach().cpu()
                delta_t_pred = delta_t_pred[0].detach().cpu()
                delta_R_pred = qua2rot_torch(delta_q_pred)
                delta_T_pred = torch.eye(4)
                delta_T_pred[:3, :3] = delta_R_pred
                delta_T_pred[:3, 3] = delta_t_pred

                # Update extrinsic: T_pred = delta_T_pred^{-1} @ T_mis
                T_pred = torch.matmul(torch.linalg.inv(delta_T_pred), T_mis.cpu())

                # Evaluate
                e_x, e_y, e_z, e_t, e_yaw, e_pitch, e_roll, e_r, _ = test_metrics(
                    T_pred.unsqueeze(0), T_velo_to_cam_gt.unsqueeze(0)
                )
                
                if stage == 3:
                    ex_epoch.append(e_x.item())
                    ey_epoch.append(e_y.item())
                    ez_epoch.append(e_z.item())
                    et_epoch.append(e_t.item())
                    eyaw_epoch.append(e_yaw.item() * 180 / torch.pi)
                    eroll_epoch.append(e_roll.item() * 180 / torch.pi)
                    epitch_epoch.append(e_pitch.item() * 180 / torch.pi)
                    er_epoch.append(e_r.item() * 180 / torch.pi)

                # Prepare for next iteration
                T_mis = T_pred.to(DEVICE)  # Update misalignment for next stage
                depth_img = gen_depth_img(pcd, T_mis, DEVICE)
                depth_img = depth_transform(depth_img)
                depth_img = depth_img.unsqueeze(0).to(DEVICE)
                
            # if i == 100:
                # break
                
    # Convert to numpy & scale
    ex_m     = np.array(ex_epoch)      # ΔX in meters
    ey_m     = np.array(ey_epoch)      # ΔY in meters
    ez_m     = np.array(ez_epoch)      # ΔZ in meters
    yaw_deg  = np.array(eyaw_epoch)    # Yaw in °
    pitch_deg= np.array(epitch_epoch)  # Pitch in °
    roll_deg = np.array(eroll_epoch)   # Roll in °

    # Plot boxplots
    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Translation residuals ---
    box_t = ax_t.boxplot(
        [ex_m, ey_m, ez_m],
        labels=['X', 'Y', 'Z'],
        showfliers=False,
        medianprops=dict(color='red'),
        patch_artist=True  # Needed to set facecolor
    )
    for patch in box_t['boxes']:
        patch.set_facecolor('white')
        patch.set_edgecolor('black')

    ax_t.set_title('Translation Error')
    ax_t.set_ylabel('Error (m)')
    ax_t.set_ylim(0, 0.03)
    ax_t.axhline(0, linestyle='--', color='gray')
    ax_t.grid(True)

    # --- Rotation residuals ---
    box_r = ax_r.boxplot(
        [roll_deg, pitch_deg, yaw_deg],
        labels=['Roll', 'Pitch', 'Yaw'],
        showfliers=False,
        medianprops=dict(color='red'),
        patch_artist=True
    )
    for patch in box_r['boxes']:
        patch.set_facecolor('white')
        patch.set_edgecolor('black')

    ax_r.set_title('Rotation Error')
    ax_r.set_ylabel('Error (°)')
    ax_r.set_ylim(0, 0.3)
    ax_r.axhline(0, linestyle='--', color='gray')
    ax_r.grid(True)

    plt.tight_layout()
    plt.savefig('error_boxplots.png', dpi=300)
    plt.close(fig)
    print("Saved boxplots to 'error_boxplots.png'")

    # Print results
    print(f"== X AXIS ==")
    print("x-axis error mean:", f"{100 * np.asarray(ex_epoch).mean():.4f}", "cm")
    print("x-axis error median:", f"{np.median(100 * np.asarray(ex_epoch)):.4f}", "cm")
    print("x-axis error std:", f"{np.std(100 * np.asarray(ex_epoch)):.4f}", "cm")

    print("== Y AXIS ==")
    print("y-axis error mean:", f"{100*np.asarray(ey_epoch).mean():.4f}", "cm")
    print("y-axis error median:", f"{np.median(100*np.asarray(ey_epoch)):.4f}", "cm")
    print("y-axis error std:", f"{np.std(100*np.asarray(ey_epoch)):.4f}", "cm")

    print("== Z AXIS ==")
    print("z-axis error mean:", f"{100*np.asarray(ez_epoch).mean():.4f}", "cm")
    print("z-axis error median:", f"{np.median(100*np.asarray(ez_epoch)):.4f}", "cm")
    print("z-axis error std:", f"{np.std(100*np.asarray(ez_epoch)):.4f}", "cm")

    print("== ET ==")
    print("Translation error:", f"{100*np.asarray(et_epoch).mean():.4f}", "cm")
    print("Translation median:", f"{np.median(100*np.asarray(et_epoch)):.4f}", "cm")
    print("Translation std:", f"{np.std(100*np.asarray(et_epoch)):.4f}", "cm")
    
    print("== ROLL ==")
    print("roll error mean:", f"{np.asarray(eroll_epoch).mean():.4f}", f"\N{DEGREE SIGN}")
    print("roll error median:", f"{np.median(np.asarray(eroll_epoch)):.4f}", f"\N{DEGREE SIGN}")
    print("roll error std:", f"{np.std(np.asarray(eroll_epoch)):.4f}", f"\N{DEGREE SIGN}")

    print("== PITCH ==")
    print("pitch error mean:", f"{np.asarray(epitch_epoch).mean():.4f}", f"\N{DEGREE SIGN}")
    print("pitch error median:", f"{np.median(np.asarray(epitch_epoch)):.4f}", f"\N{DEGREE SIGN}")
    print("pitch error std:", f"{np.std(np.asarray(epitch_epoch)):.4f}", f"\N{DEGREE SIGN}")

    print("== YAW ==")
    print("yaw error mean:", f"{np.asarray(eyaw_epoch).mean():.4f}", f"\N{DEGREE SIGN}")
    print("yaw error median:", f"{np.median(np.asarray(eyaw_epoch)):.4f}", f"\N{DEGREE SIGN}")
    print("yaw error std:", f"{np.std(np.asarray(eyaw_epoch)):.4f}", f"\N{DEGREE SIGN}")
    
    print("== ER ==")
    print("Rotation error mean:", f"{np.asarray(er_epoch).mean():.4f}", f"\N{DEGREE SIGN}")
    print("Rotation error median:", f"{np.median(np.asarray(er_epoch)):.4f}", f"\N{DEGREE SIGN}")
    print("Rotation error std:", f"{np.std(np.asarray(er_epoch)):.4f}", f"\N{DEGREE SIGN}")
    
    print("=============================================================================")
    print()

def load_models(weights, device):
    models = []
    for weight in weights:
        model = Net().to(device)
        model, _, _ = load_checkpoint(weight, model)
        model.eval()
        models.append(model)
        
    return models

if __name__ == "__main__":
    print("test start....")
    torch.multiprocessing.set_start_method('spawn')
    torch.set_printoptions(10)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    RESIZE_IMG = [350, 350]
    BATCH_SIZE = 1
    
    DEVICE = "cuda:0"
    
    rgb_transform = T.Compose([T.Resize((RESIZE_IMG[0], RESIZE_IMG[1]))])
                            #    T.Normalize(mean=[0.33, 0.36, 0.33], 
                            #                std=[0.30, 0.31, 0.32])])
    depth_transform = T.Compose([T.Resize((RESIZE_IMG[0], RESIZE_IMG[1]))])
    
    val_dataset = KITTI360_Fisheye_Dataset(rootdir="/home/rangganast/rangganast/dataset/KITTI-360",
                                        # sequences=[0, 2, 3, 4, 5, 6, 7, 9, 10],
                                        sequences=[0],
                                        split="val",
                                        camera_id="02",
                                        frame_step=1,
                                        n_scans=None,
                                        voxel_size=None,
                                        max_trans=[1.5],
                                        max_rot=[20.0],
                                        rgb_transform=rgb_transform,
                                        depth_transform=depth_transform,
                                        return_pcd=True,
                                        device=DEVICE)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=8, drop_last=True)
    
    weights = [
        'checkpoint_weights/run/LFCNet_val_1.5_20.pth.tar',
        'checkpoint_weights/run/LFCNet_val_1.0_10.pth.tar',
        'checkpoint_weights/run/LFCNet_val_0.5_5.pth.tar',
        'checkpoint_weights/run/LFCNet_val_0.2_2.pth.tar',
    ]
    
    test(
        weights,
        val_loader,
        BATCH_SIZE
    )

                

    