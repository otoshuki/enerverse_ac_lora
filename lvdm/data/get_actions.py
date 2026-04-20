import numpy as np
import os
import h5py
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter 
from lvdm.data.traj_vis_statistics import EEF2CamLeft, EEF2CamRight

def normalize_angles(radius):
    radius_normed = np.mod(radius, 2 * np.pi) - 2 * np.pi * (np.mod(radius, 2 * np.pi) > np.pi)
    return radius_normed

# def get_actions(gripper, all_ends_p=None, all_ends_o=None, slices=None, delta_act_sidx=None):

#     if delta_act_sidx is None:
#         delta_act_sidx = 1

#     if slices is None:
#         ### the first frame is repeated to fill memory
#         n = all_ends_p.shape[0]-1+delta_act_sidx
#         slices = [0,]*(delta_act_sidx-1) + list(range(all_ends_p.shape[0]))
#     else:
#         n = len(slices)

#     all_left_rpy = []
#     all_right_rpy = []
#     all_left_quat = []
#     all_right_quat = []

#     ### cam eef 30...CAM_ANGLE...
#     cvt_vis_l = Rotation.from_euler("xyz", np.array(EEF2CamLeft))
#     cvt_vis_r = Rotation.from_euler("xyz", np.array(EEF2CamRight))
#     for i in slices:
        
#         rot_l = Rotation.from_quat(all_ends_o[i, 0])
#         rot_vis_l = rot_l*cvt_vis_l
#         left_vis_quat = np.concatenate((all_ends_p[i,0], rot_vis_l.as_quat()), axis=0)
#         left_rpy = np.concatenate((all_ends_p[i,0], rot_l.as_euler("xyz", degrees=False)), axis=0)

#         rot_r = Rotation.from_quat(all_ends_o[i, 1])
#         rot_vis_r = rot_r*cvt_vis_r
#         right_vis_quat = np.concatenate((all_ends_p[i,1], rot_vis_r.as_quat()), axis=0)
#         right_rpy = np.concatenate((all_ends_p[i,1], rot_r.as_euler("xyz", degrees=False)), axis=0)

#         all_left_rpy.append(left_rpy)
#         all_right_rpy.append(right_rpy)
#         all_left_quat.append(left_vis_quat)
#         all_right_quat.append(right_vis_quat)

#     ### xyz, rpy
#     all_left_rpy = np.stack(all_left_rpy)
#     all_right_rpy = np.stack(all_right_rpy)
#     ### xyz, xyzw
#     all_left_quat = np.stack(all_left_quat)
#     all_right_quat = np.stack(all_right_quat)

#     ### xyz, xyzw, gripper
#     all_abs_actions = np.zeros([n, 16])
#     ### xyz, rpy, gripper
#     # all_delta_actions = np.zeros([n-delta_act_sidx, 14])
#     # for i in range(0, n):
#     #     all_abs_actions[i, 0:7] = all_left_quat[i, :7]
#     #     all_abs_actions[i, 7] = gripper[slices[i], 0]
#     #     all_abs_actions[i, 8:15] = all_right_quat[i, :7]
#     #     all_abs_actions[i, 15] = gripper[slices[i], 1]
#     #     if i >= delta_act_sidx:
#     #         all_delta_actions[i-delta_act_sidx, 0:6] = all_left_rpy[i, :6] - all_left_rpy[i-1, :6]
#     #         all_delta_actions[i-delta_act_sidx, 3:6] = normalize_angles(all_delta_actions[i-delta_act_sidx, 3:6])
#     #         all_delta_actions[i-delta_act_sidx, 6] = gripper[slices[i], 0] / 120.0
#     #         all_delta_actions[i-delta_act_sidx, 7:13] = all_right_rpy[i, :6] - all_right_rpy[i-1, :6]
#     #         all_delta_actions[i-delta_act_sidx, 10:13] = normalize_angles(all_delta_actions[i-delta_act_sidx, 10:13])
#     #         all_delta_actions[i-delta_act_sidx, 13] = gripper[slices[i], 1] / 120.0
    
#     ### xyz, rpy, gripper | xyz, rpy, gripper  (vel + acc per arm)
#     all_delta_actions = np.zeros([n - delta_act_sidx, 28])
    
#     for i in range(0, n):
#         all_abs_actions[i, 0:7]  = all_left_quat[i, :7]
#         all_abs_actions[i, 7]    = gripper[slices[i], 0]
#         all_abs_actions[i, 8:15] = all_right_quat[i, :7]
#         all_abs_actions[i, 15]   = gripper[slices[i], 1]
    
#         if i >= delta_act_sidx:
#             k = i - delta_act_sidx   # output row index
    
#             # ── LEFT velocity ──────────────────────────────────────────
#             all_delta_actions[k, 0:6] = all_left_rpy[i, :6] - all_left_rpy[i - 1, :6]
#             all_delta_actions[k, 3:6] = normalize_angles(all_delta_actions[k, 3:6])
#             all_delta_actions[k, 6]   = gripper[slices[i], 0] / 120.0
    
#             # ── RIGHT velocity ─────────────────────────────────────────
#             all_delta_actions[k, 14:20] = all_right_rpy[i, :6] - all_right_rpy[i - 1, :6]
#             all_delta_actions[k, 17:20] = normalize_angles(all_delta_actions[k, 17:20])
#             all_delta_actions[k, 20]    = gripper[slices[i], 1] / 120.0
    
#             # ── LEFT acceleration (zero-padded at k=0) ─────────────────
#             if k > 0:
#                 all_delta_actions[k, 7:13] = all_delta_actions[k, 0:6] - all_delta_actions[k - 1, 0:6]
#                 all_delta_actions[k, 10:13] = normalize_angles(all_delta_actions[k, 10:13])
#                 all_delta_actions[k, 13]   = all_delta_actions[k, 6] - all_delta_actions[k - 1, 6]
    
#             # ── RIGHT acceleration (zero-padded at k=0) ────────────────
#             if k > 0:
#                 all_delta_actions[k, 21:27] = all_delta_actions[k, 14:20] - all_delta_actions[k - 1, 14:20]
#                 all_delta_actions[k, 24:27] = normalize_angles(all_delta_actions[k, 24:27])
#                 all_delta_actions[k, 27]   = all_delta_actions[k, 20] - all_delta_actions[k - 1, 20]

#     return all_abs_actions, all_delta_actions

def get_actions(gripper, all_ends_p=None, all_ends_o=None, slices=None, delta_act_sidx=None):

    if delta_act_sidx is None:
        delta_act_sidx = 1

    if slices is None:
        #the_first_frame_is_repeated_to_fill_memory
        n = all_ends_p.shape[0] - 1 + delta_act_sidx
        slices = [0,] * (delta_act_sidx - 1) + list(range(all_ends_p.shape[0]))
    else:
        n = len(slices)

    all_left_rpy = []
    all_right_rpy = []
    all_left_quat = []
    all_right_quat = []

    #cam_eef_30...CAM_ANGLE...
    cvt_vis_l = Rotation.from_euler("xyz", np.array(EEF2CamLeft))
    cvt_vis_r = Rotation.from_euler("xyz", np.array(EEF2CamRight))
    for i in slices:
        rot_l = Rotation.from_quat(all_ends_o[i, 0])
        rot_vis_l = rot_l * cvt_vis_l
        left_vis_quat = np.concatenate((all_ends_p[i,0], rot_vis_l.as_quat()), axis=0)
        left_rpy = np.concatenate((all_ends_p[i,0], rot_l.as_euler("xyz", degrees=False)), axis=0)

        rot_r = Rotation.from_quat(all_ends_o[i, 1])
        rot_vis_r = rot_r * cvt_vis_r
        right_vis_quat = np.concatenate((all_ends_p[i,1], rot_vis_r.as_quat()), axis=0)
        right_rpy = np.concatenate((all_ends_p[i,1], rot_r.as_euler("xyz", degrees=False)), axis=0)

        all_left_rpy.append(left_rpy)
        all_right_rpy.append(right_rpy)
        all_left_quat.append(left_vis_quat)
        all_right_quat.append(right_vis_quat)

    #xyz,_rpy
    all_left_rpy = np.stack(all_left_rpy)
    all_right_rpy = np.stack(all_right_rpy)
    #xyz,_xyzw
    all_left_quat = np.stack(all_left_quat)
    all_right_quat = np.stack(all_right_quat)

    #vectorized_absolute_actions_(replaces_the_first_part_of_your_loop)
    all_abs_actions = np.zeros([n, 16])
    all_abs_actions[:, 0:7] = all_left_quat[:, :7]
    all_abs_actions[:, 7] = gripper[slices, 0]
    all_abs_actions[:, 8:15] = all_right_quat[:, :7]
    all_abs_actions[:, 15] = gripper[slices, 1]

    #xyz,_rpy,_gripper_|_xyz,_rpy,_gripper__(vel_+_acc_per_arm)
    all_delta_actions = np.zeros([n - delta_act_sidx, 28])
    
    if n > delta_act_sidx:
        #1._vectorized_extraction_of_raw_velocities
        raw_left_vel = all_left_rpy[delta_act_sidx:] - all_left_rpy[delta_act_sidx - 1 : -1]
        raw_left_vel[:, 3:6] = normalize_angles(raw_left_vel[:, 3:6])
        
        raw_right_vel = all_right_rpy[delta_act_sidx:] - all_right_rpy[delta_act_sidx - 1 : -1]
        raw_right_vel[:, 3:6] = normalize_angles(raw_right_vel[:, 3:6])
        
        #2._apply_savitzky-golay_filter_to_smooth_the_velocities
        #window_length=5_and_polyorder=2_are_excellent_defaults_for_30Hz_kinematics
        wl = min(5, raw_left_vel.shape[0] if raw_left_vel.shape[0] % 2 != 0 else raw_left_vel.shape[0] - 1)
        
        if wl > 2:
            smooth_l_vel = savgol_filter(raw_left_vel, window_length=wl, polyorder=2, axis=0)
            smooth_r_vel = savgol_filter(raw_right_vel, window_length=wl, polyorder=2, axis=0)
        else:
            smooth_l_vel = raw_left_vel
            smooth_r_vel = raw_right_vel
            
        #re-normalize_angles_after_smoothing_polynomial_fit
        smooth_l_vel[:, 3:6] = normalize_angles(smooth_l_vel[:, 3:6])
        smooth_r_vel[:, 3:6] = normalize_angles(smooth_r_vel[:, 3:6])
        
        #3._assign_smoothed_velocities_to_all_delta_actions
        all_delta_actions[:, 0:6] = smooth_l_vel
        all_delta_actions[:, 6] = gripper[slices[delta_act_sidx:], 0] / 120.0
        
        all_delta_actions[:, 14:20] = smooth_r_vel
        all_delta_actions[:, 20] = gripper[slices[delta_act_sidx:], 1] / 120.0
        
        #4._vectorized_extraction_of_accelerations_from_smoothed_velocities
        if all_delta_actions.shape[0] > 1:
            #left_acceleration
            left_acc = smooth_l_vel[1:] - smooth_l_vel[:-1]
            left_acc[:, 3:6] = normalize_angles(left_acc[:, 3:6])
            all_delta_actions[1:, 7:13] = left_acc
            all_delta_actions[1:, 13] = all_delta_actions[1:, 6] - all_delta_actions[:-1, 6]
            
            #right_acceleration
            right_acc = smooth_r_vel[1:] - smooth_r_vel[:-1]
            right_acc[:, 3:6] = normalize_angles(right_acc[:, 3:6])
            all_delta_actions[1:, 21:27] = right_acc
            all_delta_actions[1:, 27] = all_delta_actions[1:, 20] - all_delta_actions[:-1, 20]

    return all_abs_actions, all_delta_actions

def parse_h5(h5_file, slices=None, delta_act_sidx=1):
    """
    read and parse .h5 file, and obtain the absolute actions and the action differences
    """
    with h5py.File(h5_file, "r") as fid:
        all_abs_gripper = np.array(fid[f"state/effector/position"], dtype=np.float32)
        all_ends_p = np.array(fid["state/end/position"], dtype=np.float32)
        all_ends_o = np.array(fid["state/end/orientation"], dtype=np.float32)

    all_abs_actions, all_delta_actions = get_actions(
        gripper=all_abs_gripper,
        slices=slices,
        delta_act_sidx=delta_act_sidx,
        all_ends_p=all_ends_p,
        all_ends_o=all_ends_o,
    )
    return all_abs_actions, all_delta_actions