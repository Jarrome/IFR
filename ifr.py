import torch
import copy
import numpy as np
from random import sample
import open3d as o3d
import scipy.spatial

import se_math.se3 as se3
import se_math.invmat as invmat

import utils


from time import time
import pdb

def IRLS_p_torch(y,X,maxiter, w, IRLS_p = 1, d=0.0001, tolerance=1e-3):
    n,p = X.shape
    delta = torch.ones((1,n),dtype=torch.float64).to(X) * d
    if w is None:
        w = torch.ones((1,n),dtype=torch.float64).to(X)
    #W = torch.diag(w[0,:]) # n,n
    #XTW = X.transpose(0,1).matmul(W)
    XTW = X.transpose(0,1)*w #p,n
    B = XTW.matmul(X).inverse().matmul(XTW.matmul(y))
    for _ in range(maxiter):
        _B = B
        _w = torch.abs(y-X.matmul(B)).transpose(0,1)
        #w = 1./torch.max(delta,_w)
        w = torch.max(delta,_w) ** (IRLS_p-2)
        #W = torch.diag(w[0,:])
        #XTW = X.transpose(0,1).matmul(W)
        XTW = X.transpose(0,1)*w
        B = XTW.matmul(X).inverse().matmul(XTW.matmul(y))
        tol = torch.abs(B-_B).sum()
        if tol < tolerance:
            return B, w
    return B, w
       
    
'''
    feed in with point cloud pc1 and pc2

'''

class IFR:
    def __init__(self,scale=1,maxiter = 20, norm_radius=1., zero_mean = False, pa_num = 1000, trunc=True, rand_pa=True,encoder_id=1, kp_nb=False,\
            # IRLS parameters
            use_IRLS=True,IRLS_p=1, IRLS_d=1e-2, IRLS_maxiter=1000, IRLS_reuse_w=True\
            ):
        # torch
        self.device = "cpu"

        # hyper param
        self.maxiter = maxiter
        self.xtol = 1e-6#1e-6 # threhold to early stop
        self.scale = scale

        self.norm_radius = norm_radius


        self.encoder_id = encoder_id
        self.zero_mean = zero_mean # zero_mean before process

        self.pa_num = pa_num
        self.trunc = trunc

        self.rand_pa= rand_pa
        self.kp_nb = kp_nb
        if self.rand_pa:
            self.pa_type = 'rand'
        else:
            if self.kp_nb:
                self.pa_type = 'kp_neighbor'
            else:
                self.pa_type = 'neighbor'
        # IRLS
        self.use_IRLS=use_IRLS
        self.IRLS_p = IRLS_p
        self.IRLS_d = IRLS_d
        self.IRLS_maxiter = IRLS_maxiter
        self.IRLS_reuse_w = IRLS_reuse_w


        if type(scale) == int:
            #self.sigma = self.scale * self.sigma
            pass
        else: # scale is something like [100,100,50]
            #self.sigma = self.scale[0] * self.sigma
            self.scale = np.array([[self.scale]])



        # functions
        self.inverse = invmat.InvMatrix.apply
        self.exp = se3.Exp  # [B, 6] -> [B, 4, 4]
        self.transform = se3.transform  # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]


        # tree
        self.pc0_tree = None
        self.pc1_tree = None
    def reset_pa(self):
        '''
            self.pa: 1,self.pa_num,3
        '''

        if self.pa_type == "neighbor":
            sc = self.scale
            perm = torch.randperm(self.p0.size(0))

            while perm.shape[0] < self.pa_num:
                perm1 = torch.randperm(self.p0.size(0))
                perm = torch.cat([perm, perm1[:self.pa_num-self.p0.size(0)]],axis=0).to(self.device)

            try:
                pa = self.p0[perm[:self.pa_num],:] + torch.normal(0,sc/5,size=[self.pa_num]).unsqueeze(1).to(self.device)*self.p0_ns[perm[:self.pa_num],:]
            except:
                pdb.set_trace()
            self.pa = pa.unsqueeze(0)

        elif self.pa_type == 'rand':
            # random pa
            pa_np = (np.random.random((1,self.pa_num,3))-.5)*2 * self.scale
            self.pa = torch.tensor(pa_np,dtype=torch.float32).to(self.device)
        elif self.pa_type == 'kp_neighbor':
            sc = self.scale

            nbhood = np.random.normal( size=(self.pa_num,3)) * sc/5
            pa_np = self.keypoints0[np.random.choice(self.keypoints0.shape[0],self.pa_num,replace=True),:] + nbhood
            self.pa = torch.tensor(pa_np,dtype=torch.float32).to(self.device).unsqueeze(0)
            
    def open3d_pc(self,p0):
        '''
            p0:N,3
        '''
        p0_src = o3d.geometry.PointCloud()
        p0_src.points = o3d.utility.Vector3dVector(p0[:,:])
        #p0_src = p0_src_.voxel_down_sample(voxel_size=0.1)

        p0_src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.norm_radius,
                                                      max_nn=30))
        p0_ns = np.asarray(p0_src.normals)

        pcd_tree = scipy.spatial.cKDTree(np.asarray(p0_src.points))
        return p0_src, pcd_tree, p0_ns

    def prepare(self,p0, p1):
        '''
            p0, p1: N(M),3
        '''
        if type(p0) != torch.Tensor:
            p0 = torch.tensor(p0).unsqueeze(0)
        if type(p1) != torch.Tensor:
            p1 = torch.tensor(p1).unsqueeze(0)

        if self.zero_mean:
            a0 = torch.eye(4).view(1, 4, 4).to(p0)  # [1, 4, 4]
            a1 = torch.eye(4).view(1, 4, 4).to(p1)  # [1, 4, 4]

            p0_m = p0.mean(1)#((p0.max(1)[0]-p0.min(1)[0])/2)#p0.mean(1)
            a0[:, 0:3, 3] = p0_m   # global frame
            p0 = p0 - p0_m.unsqueeze(1)   # local frame

            p1_m = p1.mean(1)#((p1.max(1)[0]-p1.min(1)[0])/2)#p1.mean(1)
            a1[:, 0:3, 3] = -p1_m   # global frame
            p1 = p1 - p1_m.unsqueeze(1)   # local frame

            self.a0 = a0
            self.a1 = a1


        if type(p0) == torch.Tensor:
            p0 = p0.cpu().detach().numpy()[0,:,:]
        if type(p1) == torch.Tensor:
            p1 = p1.cpu().detach().numpy()[0,:,:]

        # 1. generate kd-tree for pc1 and pc2
        pc0, self.pc0_tree, p0_ns = self.open3d_pc(p0)
        if self.kp_nb:
            self.keypoints0 = np.asarray(o3d.geometry.keypoint.compute_iss_keypoints(pc0, gamma_21=.5, gamma_32=.5,min_neighbors=10).points)


        pc1, self.pc1_tree, p1_ns = self.open3d_pc(p1)
        self.p0 = torch.tensor(np.asarray(pc0.points),dtype=torch.float32).to(self.device)
        self.p1 = torch.tensor(np.asarray(pc1.points),dtype=torch.float32).to(self.device)
        self.p0_ns = torch.tensor(p0_ns,dtype=torch.float32).to(self.device)
        self.p1_ns = torch.tensor(p1_ns,dtype=torch.float32).to(self.device)



    def encoder(self, p1, pc1_tree, p1_ns, pa):
        if self.encoder_id == 1:
            return self.encoder_v1(p1, pc1_tree, p1_ns, pa)
        else:
            return self.encoder_v4(p1, pc1_tree, p1_ns, pa)


    def encoder_v1(self, p1, pc1_tree, p1_ns, pa):
        '''
            pa: L,3
            p1: N,3
        '''
        assert False, "This v1 encoder is not used"


    def encoder_v4(self, p1, pc1_tree, p1_ns, pa):
        '''
            pa: L,3
            p1: N,3
        '''
        pa_np = pa[0,:,:].cpu().detach().numpy()
        dist_min, dist_min_id = pc1_tree.query(pa_np) # L

        # turn to torch
        dist_min_id = torch.tensor(dist_min_id).to(self.device)
        diff_min = p1[dist_min_id,:] - pa[0,:,:] #L,3

        dist_min = torch.sum(diff_min**2,dim=1) #L


        '''
            euclidean dist
        '''
        df = dist_min
        grad = - (-2*diff_min)

        L = grad.shape[0]
        # grad should be L,L,3
        try:
            grad = grad.unsqueeze(0) * torch.eye(L).unsqueeze(-1).to(self.device) #1,L,3 * L,L,1 -> L,L,3
        except Exception as e:
            pdb.set_trace()
            print(e)
        
        # for trunct=False
        #return df.unsqueeze(0), (None,None), grad.unsqueeze(0)


        pa_dirc = diff_min / torch.sqrt(dist_min.unsqueeze(-1))#L,3


        _, map_idx, counts = dist_min_id.unique(return_counts=True,return_inverse=True)
        bad = counts > 3
        mask = bad[map_idx]


        return df.unsqueeze(0), (mask.unsqueeze(0), pa_dirc.unsqueeze(0)), grad.unsqueeze(0)

    ''' -----------------------------------------------------------------------------------------'''
        
    # update the transformation
    def update(self, g, dx):
        # [B, 4, 4] x [B, 6] -> [B, 4, 4]
        dg = self.exp(dx)
        return dg.matmul(g)

 
    def cal_Jac(self, pa, grad_f0_pa):
        batch_size = pa.shape[0]
        # 1. get "warp Jacobian", warp => Identity matrix, can be pre-computed
        # grad_f0_pa is B, L, L, 3
        g_ = torch.zeros(batch_size, 6).to(pa)
        warp_jac = utils.compute_warp_jac(g_, pa, num_points=pa.shape[1])   # B x L x 3 x 6
        #J = torch.einsum('iajk,ijkm->iam', grad_f0_pa, warp_jac) #B,L,6
        # fast alternative
        grad_f0_pa_ = grad_f0_pa.sum(axis=1)#B,L,3
        interm = grad_f0_pa_.unsqueeze(-1)*warp_jac # B,L,3,6
        J = interm.sum(2)

        return J


    # compute pinv(j) to solve j*x = -r
    def solve_g_origin(self, j_in, r_in, w, mask, trunc=True):
        if trunc:
            j = j_in[:,mask[0,:],:]
            r = r_in[:,mask[0,:]]
        else:
            j = j_in
            r = r_in
        jt = j.transpose(1, 2)  # [b, 6, k]
        j = j#*mask.unsqueeze(-1)
        h = jt.bmm(j)  # [b, 6, 6]
        # h = h + u_lamda * identity
        b = self.inverse(h)
        #b = torch.pinverse(h)
        pinv = b.bmm(jt)  # [b, 6, k]
        pinv = pinv# * mask.unsqueeze(1)

        dx = -pinv.bmm(r.unsqueeze(-1)).view(1, 6)

        return dx,w



    # compute with IRLS
    def solve_g(self, j_in, r_in, w, mask, trunc=True):
        if trunc:
            j = j_in[:,mask[0,:],:]
            r = r_in[:,mask[0,:]]
            w_ = w[:,mask[0,:]]



            dx,w_ = IRLS_p_torch(r[0,:].unsqueeze(-1),j[0,:,:],IRLS_p=self.IRLS_p, d=self.IRLS_d,maxiter=self.IRLS_maxiter,w=w_)
            w[mask] = w_

            dx = -dx.view(1,6)

        else:
            dx,w = IRLS_p_torch(r_in[0,:].unsqueeze(-1),j_in[0,:,:],IRLS_p=self.IRLS_p, d=self.IRLS_d,maxiter=self.IRLS_maxiter,w=w if self.IRLS_reuse_w else None)

            dx = -dx.view(1,6)

        return dx,w



    def solve(self):
        g = torch.eye(4).to(self.device).view(1, 4, 4).expand(1, 4, 4).contiguous()
        B,L,_ = self.pa.shape

        f0, mask_f0, _ = self.encoder(self.p0, self.pc0_tree, self.p0_ns, self.pa)
        w_f = torch.ones(f0.shape,dtype=torch.float64).to(f0)
        for itr in range(self.maxiter):
            # 2.1 Jacobian
            pa_ = self.transform(g.unsqueeze(1), self.pa)# [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
            f1, mask_f1, f1_grad = self.encoder(self.p1,self.pc1_tree,self.p1_ns,pa_)
            J = self.cal_Jac(pa_, f1_grad)

            r = f1-f0
            if self.trunc:
                if self.encoder_id == 1:
                    mask_f0_, pa_dir0 = mask_f0
                    mask_f1_, pa_dir1 = mask_f1
                    direction_T = torch.clone(g)
                    direction_T[:,:3,3] = 0
                    pa_dir1 = self.transform(direction_T.transpose(1,2).unsqueeze(1), pa_dir1)
                    mask_direct = ( pa_dir0 * pa_dir1 ).sum(2) # 1,N

                    mask = torch.zeros(1)
                    th = 1
                    while mask.sum() < 10:
                        th -= .1
                        mask = (mask_f0>th)*(mask_f1>th) * (mask_direct>th)


                elif self.encoder_id == 4:
                    mask_f0_, pa_dir0 = mask_f0
                    mask_f1_, pa_dir1 = mask_f1
                    direction_T = torch.clone(g)
                    direction_T[:,:3,3] = 0
                    pa_dir1 = self.transform(direction_T.transpose(1,2).unsqueeze(1), pa_dir1)
                    mask_direct = ( pa_dir0 * pa_dir1 ).sum(2) # 1,N

                    mask = torch.zeros(1)
                    th = 1
                    while mask.sum() < 10:
                        th -= .1
                        mask_ = (mask_direct) > th
                        if th < .1:
                            mask = mask_
                        else:
                            mask = mask_ * ~mask_f0_ * ~mask_f1_
            else:
                mask = None

            if self.use_IRLS:
                dx,w_f = self.solve_g(J,r,mask=mask,trunc=self.trunc,w=w_f)
            else:
                dx,w_f = self.solve_g_origin(J,r,mask=mask,trunc=self.trunc,w=w_f)


            # 2.2 update
            check = dx.norm(p=2, dim=1, keepdim=True).max()
            #print(check)

            if float(check) < self.xtol:
                if itr == 0:
                    self.last_err = 0  # no update.
                break

            g = self.update(g, dx)

        g = se3.inverse(g)
        return g

 


    def register(self,pc1,pc2):
        self.prepare(pc1,pc2)
        self.reset_pa()
        g = self.solve()
        if self.zero_mean:
            g = self.a0.to(g).bmm(g)
            g = g.bmm(self.a1.to(g))
        return g.cpu().detach().numpy()


