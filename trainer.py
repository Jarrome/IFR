""" part of source code from PointNetLK (https://github.com/hmgoforth/PointNetLK), modified. """

import torch
import numpy as np
from scipy.spatial.transform import Rotation
import tqdm
import logging
import open3d as o3d
#from open3d.web_visualizer import draw   # for notebook
from ifr import IFR

import scipy

import utils
import pdb

from time import time


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class TrainerAnalyticalPointNetLK:
    def __init__(self, args):
        # PointNet
        self.dim_k = args.dim_k
        # LK
        self.device = args.device
        self.max_iter = args.max_iter
        self.xtol = 1.0e-7
        self.p0_zero_mean = True
        self.p1_zero_mean = True
        # network
        self.embedding = args.embedding
        self.filename = args.outfile

        self.dataset_type = args.dataset_type
        
    def create_model(self):
        if self.dataset_type=='modelnet':
            return IFR(scale=.5,maxiter=10,zero_mean=True, trunc=False, rand_pa=True,encoder_id=4)
        elif self.dataset_type=='3dmatch':
            return IFR(scale=2, maxiter=20,zero_mean=True,trunc=True,rand_pa=False, kp_nb=False,encoder_id=4)
        elif self.dataset_type=='shapenet2':
            return IFR(scale=1,maxiter=10,zero_mean=True, trunc=False, rand_pa=True,encoder_id=2)
        elif self.dataset_type=='stanford':
            return IFR(scale=1,maxiter=10,zero_mean=True, trunc=False, rand_pa=True,encoder_id=2)




    def test_one_epoch(self, ifr, testloader, device, mode, data_type='synthetic', vis=False, toyexample=False):
        rotations_gt = []
        translation_gt = []
        rotations_ab = []
        translation_ab = []
        

        
        for i, data in tqdm.tqdm(enumerate(testloader), total=len(testloader), ncols=73, leave=False):
            # if voxelization: VxNx3, Vx3, 1x4x4
            if data_type == 'real':
                if vis:
                    voxel_features_p0, voxel_coords_p0, voxel_features_p1, voxel_coords_p1, gt_pose, p0, p1 = data
                    p0 = p0.float().to(device)
                    p1 = p1.float().to(device)
                else:
                    voxel_features_p0, voxel_coords_p0, voxel_features_p1, voxel_coords_p1, gt_pose = data
                voxel_features_p0 = voxel_features_p0.reshape(-1, voxel_features_p0.shape[2], voxel_features_p0.shape[3]).to(device)
                voxel_features_p1 = voxel_features_p1.reshape(-1, voxel_features_p1.shape[2], voxel_features_p1.shape[3]).to(device)
                voxel_coords_p0 = voxel_coords_p0.reshape(-1, voxel_coords_p0.shape[2]).to(device)
                voxel_coords_p1 = voxel_coords_p1.reshape(-1, voxel_coords_p1.shape[2]).to(device)
                gt_pose = gt_pose.float().to(device)
                # estimate
                estimated_pose = ifr.register_voxel(voxel_features_p0, voxel_coords_p0, voxel_features_p1, voxel_coords_p1)


            else:
                p0, p1, gt_pose = data
                p0 = p0.float().to(device)
                p1 = p1.float().to(device)

                if self.dataset_type=='3dmatch':
                    p0_np = p0.cpu().detach().numpy()[0,:,:]
                    p1_np = p1.cpu().detach().numpy()[0,:,:]

                    estimated_pose = ifr.register(p0, p1)
                else:
                    '''
                    p0, p1, gt_pose = data
                    p0 = p0.float().to(device)
                    p1 = p1.float().to(device)
                    '''
                    estimated_pose = ifr.register(p0, p1)





            ig_gt = gt_pose.cpu().contiguous().view(-1, 4, 4) # --> [1, 4, 4]
            g_hat = torch.tensor(estimated_pose).float()#.cpu().contiguous().view(-1, 4, 4).detach() # --> [1, 4, 4], p1->p0 (S->T)

            dg = g_hat.bmm(ig_gt)   # if correct, dg == identity matrix.
            dx = utils.log(dg)   # --> [1, 6] (if corerct, dx == zero vector)
            dn = dx.norm(p=2, dim=1)   # --> [1]
            dm = dn.mean()
            
            LOGGER.info('test, %d/%d, %d iterations, %f', i, len(testloader), 20, dm)

                    
            # euler representation for ground truth
            tform_gt = ig_gt.squeeze().numpy().transpose()
            R_gt = tform_gt[:3, :3]
            euler_angle = Rotation.from_matrix(R_gt)
            anglez_gt, angley_gt, anglex_gt = euler_angle.as_euler('zyx')
            angle_gt = np.array([anglex_gt, angley_gt, anglez_gt])
            rotations_gt.append(angle_gt)
            trans_gt_t = -R_gt.dot(tform_gt[3, :3])
            translation_gt.append(trans_gt_t)
            # euler representation for predicted transformation
            tform_ab = g_hat.squeeze().numpy()
            R_ab = tform_ab[:3, :3]
            euler_angle = Rotation.from_matrix(R_ab)
            anglez_ab, angley_ab, anglex_ab = euler_angle.as_euler('zyx')
            angle_ab = np.array([anglex_ab, angley_ab, anglez_ab])
            rotations_ab.append(angle_ab)
            trans_ab = tform_ab[:3, 3]
            translation_ab.append(trans_ab)
        with open(self.filename,'wb') as f:
            np.save(f,np.stack(rotations_gt))
            np.save(f,np.stack(translation_gt))
            np.save(f,np.stack(rotations_ab))
            np.save(f,np.stack(translation_ab))





        utils.test_metrics(rotations_gt, translation_gt, rotations_ab, translation_ab, self.filename)
        
        return 

    def compute_loss(self, ptnetlk, data, device, mode, data_type='synthetic', num_random_points=100):
        # 1. non-voxelization
        if data_type == 'synthetic':
            p0, p1, gt_pose = data
            p0 = p0.to(self.device)
            p1 = p1.to(self.device)
            gt_pose = gt_pose.to(device)
            r = model.AnalyticalPointNetLK.do_forward(ptnetlk, p0, None,
                                p1, None, self.max_iter, self.xtol, self.p0_zero_mean, self.p1_zero_mean, mode, data_type, num_random_points)
        else:
            # 2. voxelization
            voxel_features_p0, voxel_coords_p0, voxel_features_p1, voxel_coords_p1, gt_pose = data
            voxel_features_p0 = voxel_features_p0.reshape(-1, voxel_features_p0.shape[2], voxel_features_p0.shape[3]).to(device)
            voxel_features_p1 = voxel_features_p1.reshape(-1, voxel_features_p1.shape[2], voxel_features_p1.shape[3]).to(device)
            voxel_coords_p0 = voxel_coords_p0.reshape(-1, voxel_coords_p0.shape[2]).to(device)
            voxel_coords_p1 = voxel_coords_p1.reshape(-1, voxel_coords_p1.shape[2]).to(device)
            gt_pose = gt_pose.reshape(-1, gt_pose.shape[2], gt_pose.shape[3]).to(device)
            
            r = model.AnalyticalPointNetLK.do_forward(ptnetlk, voxel_features_p0_, voxel_coords_p0_,
                    voxel_features_p1_, voxel_coords_p1_, self.max_iter, self.xtol, self.p0_zero_mean, self.p1_zero_mean, mode, data_type, num_random_points)

        estimated_pose = ptnetlk.g

        loss_pose = model.AnalyticalPointNetLK.comp(estimated_pose, gt_pose)
        pr = ptnetlk.prev_r
        if pr is not None:
            loss_r = model.AnalyticalPointNetLK.rsq(r - pr)
        else:
            loss_r = model.AnalyticalPointNetLK.rsq(r)
        loss = loss_r + loss_pose

        return loss, loss_pose

