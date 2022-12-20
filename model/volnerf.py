import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict

import lpips
from external.pohsun_ssim import pytorch_ssim

import util,util_vis
from util import log,debug
# from . import base
from . import nerf as NeRF
import camera

import tinycudann as tcnn

# ============================ main engine for training and evaluation ============================

class Model(NeRF.Model):

    def __init__(self,opt):
        super().__init__(opt)
        self.lpips_loss = lpips.LPIPS(net="alex").to(opt.device)

    def load_dataset(self,opt,eval_split="val"):
        super().load_dataset(opt,eval_split=eval_split)
        # prefetch all training data
        self.train_data.prefetch_all_data(opt)
        self.train_data.all = edict(util.move_to_device(self.train_data.all,opt.device))

    def setup_optimizer(self,opt):
        log.info("setting up optimizers...")
        optimizer = getattr(torch.optim,opt.optim.algo)
        self.optim = optimizer([dict(params=self.graph.nerf.parameters(),lr=opt.optim.lr)])
        if opt.nerf.fine_sampling:
            self.optim.add_param_group(dict(params=self.graph.nerf_fine.parameters(),lr=opt.optim.lr))
        # set up scheduler
        if opt.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler,opt.optim.sched.type)
            if opt.optim.lr_end:
                assert(opt.optim.sched.type=="ExponentialLR")
                opt.optim.sched.gamma = (opt.optim.lr_end/opt.optim.lr)**(1./opt.max_iter)
            kwargs = { k:v for k,v in opt.optim.sched.items() if k!="type" }
            self.sched = scheduler(self.optim,**kwargs)

    def train(self,opt):
        # before training
        log.title("TRAINING START")
        self.timer = edict(start=time.time(),it_mean=None)
        self.graph.train()
        self.ep = 0 # dummy for timer
        # training
        if self.iter_start==0: self.validate(opt,0)
        loader = tqdm.trange(opt.max_iter,desc="training",leave=False)
        for self.it in loader:
            if self.it<self.iter_start: continue
            # set var to all available images
            var = self.train_data.all
            self.train_iteration(opt,var,loader)
            if opt.optim.sched: self.sched.step()
            if self.it%opt.freq.val==0: self.validate(opt,self.it)
            if self.it%opt.freq.ckpt==0: self.save_checkpoint(opt,ep=None,it=self.it)
        # after training
        if opt.tb:
            self.tb.flush()
            self.tb.close()
        if opt.visdom: self.vis.close()
        log.title("TRAINING DONE")

    @torch.no_grad()
    def log_scalars(self,opt,var,loss,metric=None,step=0,split="train"):
        super().log_scalars(opt,var,loss,metric=metric,step=step,split=split)
        # log learning rate
        if split=="train":
            lr = self.optim.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split,"lr"),lr,step)
            if opt.nerf.fine_sampling:
                lr = self.optim.param_groups[1]["lr"]
                self.tb.add_scalar("{0}/{1}".format(split,"lr_fine"),lr,step)
        # compute PSNR
        psnr = -10*loss.render.log10()
        self.tb.add_scalar("{0}/{1}".format(split,"PSNR"),psnr,step)
        if opt.nerf.fine_sampling:
            psnr = -10*loss.render_fine.log10()
            self.tb.add_scalar("{0}/{1}".format(split,"PSNR_fine"),psnr,step)

    @torch.no_grad()
    def visualize(self,opt,var,step=0,split="train",eps=1e-10):
        if opt.tb:
            util_vis.tb_image(opt,self.tb,step,split,"image",var.image)
            if not opt.nerf.rand_rays or split!="train":
                invdepth = (1-var.depth)/var.opacity if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
                rgb_map = var.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
                invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                util_vis.tb_image(opt,self.tb,step,split,"rgb",rgb_map)
                util_vis.tb_image(opt,self.tb,step,split,"invdepth",invdepth_map)
                if opt.nerf.fine_sampling:
                    invdepth = (1-var.depth_fine)/var.opacity_fine if opt.camera.ndc else 1/(var.depth_fine/var.opacity_fine+eps)
                    rgb_map = var.rgb_fine.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
                    invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                    util_vis.tb_image(opt,self.tb,step,split,"rgb_fine",rgb_map)
                    util_vis.tb_image(opt,self.tb,step,split,"invdepth_fine",invdepth_map)

    @torch.no_grad()
    def get_all_training_poses(self,opt):
        # get ground-truth (canonical) camera poses
        pose_GT = self.train_data.get_all_camera_poses(opt).to(opt.device)
        return None,pose_GT

    @torch.no_grad()
    def evaluate_full(self,opt,eps=1e-10):
        self.graph.eval()
        loader = tqdm.tqdm(self.test_loader,desc="evaluating",leave=False)
        res = []
        test_path = "{}/test_view".format(opt.output_path)
        os.makedirs(test_path,exist_ok=True)
        for i,batch in enumerate(loader):
            var = edict(batch)
            var = util.move_to_device(var,opt.device)
            if opt.model=="barf" and opt.optim.test_photo:
                # run test-time optimization to factorize imperfection in optimized poses from view synthesis evaluation
                var = self.evaluate_test_time_photometric_optim(opt,var)
            var = self.graph.forward(opt,var,mode="eval")
            # evaluate view synthesis
            invdepth = (1-var.depth)/var.opacity if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
            rgb_map = var.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
            invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
            psnr = -10*self.graph.MSE_loss(rgb_map,var.image).log10().item()
            ssim = pytorch_ssim.ssim(rgb_map,var.image).item()
            lpips = self.lpips_loss(rgb_map*2-1,var.image*2-1).item()
            res.append(edict(psnr=psnr,ssim=ssim,lpips=lpips))
            # dump novel views
            torchvision_F.to_pil_image(rgb_map.cpu()[0]).save("{}/rgb_{}.png".format(test_path,i))
            torchvision_F.to_pil_image(var.image.cpu()[0]).save("{}/rgb_GT_{}.png".format(test_path,i))
            torchvision_F.to_pil_image(invdepth_map.cpu()[0]).save("{}/depth_{}.png".format(test_path,i))
        # show results in terminal
        print("--------------------------")
        print("PSNR:  {:8.2f}".format(np.mean([r.psnr for r in res])))
        print("SSIM:  {:8.2f}".format(np.mean([r.ssim for r in res])))
        print("LPIPS: {:8.2f}".format(np.mean([r.lpips for r in res])))
        print("--------------------------")
        # dump numbers to file
        quant_fname = "{}/quant.txt".format(opt.output_path)
        with open(quant_fname,"w") as file:
            for i,r in enumerate(res):
                file.write("{} {} {} {}\n".format(i,r.psnr,r.ssim,r.lpips))

    @torch.no_grad()
    def generate_videos_synthesis(self,opt,eps=1e-10):
        self.graph.eval()
        if opt.data.dataset=="blender":
            test_path = "{}/test_view".format(opt.output_path)
            # assume the test view synthesis are already generated
            print("writing videos...")
            rgb_vid_fname = "{}/test_view_rgb.mp4".format(opt.output_path)
            depth_vid_fname = "{}/test_view_depth.mp4".format(opt.output_path)
            os.system("ffmpeg -y -framerate 30 -i {0}/rgb_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(test_path,rgb_vid_fname))
            os.system("ffmpeg -y -framerate 30 -i {0}/depth_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(test_path,depth_vid_fname))
        else:
            pose_pred,pose_GT = self.get_all_training_poses(opt)
            poses = pose_pred if opt.model=="barf" else pose_GT
            if opt.model=="barf" and opt.data.dataset=="llff":
                _,sim3 = self.prealign_cameras(opt,pose_pred,pose_GT)
                scale = sim3.s1/sim3.s0
            else: scale = 1
            # rotate novel views around the "center" camera of all poses
            idx_center = (poses-poses.mean(dim=0,keepdim=True))[...,3].norm(dim=-1).argmin()
            pose_novel = camera.get_novel_view_poses(opt,poses[idx_center],N=60,scale=scale).to(opt.device)
            # render the novel views
            novel_path = "{}/novel_view".format(opt.output_path)
            os.makedirs(novel_path,exist_ok=True)
            pose_novel_tqdm = tqdm.tqdm(pose_novel,desc="rendering novel views",leave=False)
            intr = edict(next(iter(self.test_loader))).intr[:1].to(opt.device) # grab intrinsics
            for i,pose in enumerate(pose_novel_tqdm):
                ret = self.graph.render_by_slices(opt,pose[None],intr=intr) if opt.nerf.rand_rays else \
                      self.graph.render(opt,pose[None],intr=intr)
                invdepth = (1-ret.depth)/ret.opacity if opt.camera.ndc else 1/(ret.depth/ret.opacity+eps)
                rgb_map = ret.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
                invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                torchvision_F.to_pil_image(rgb_map.cpu()[0]).save("{}/rgb_{}.png".format(novel_path,i))
                torchvision_F.to_pil_image(invdepth_map.cpu()[0]).save("{}/depth_{}.png".format(novel_path,i))
            # write videos
            print("writing videos...")
            rgb_vid_fname = "{}/novel_view_rgb.mp4".format(opt.output_path)
            depth_vid_fname = "{}/novel_view_depth.mp4".format(opt.output_path)
            os.system("ffmpeg -y -framerate 30 -i {0}/rgb_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(novel_path,rgb_vid_fname))
            os.system("ffmpeg -y -framerate 30 -i {0}/depth_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(novel_path,depth_vid_fname))

# ============================ computation graph for forward/backprop ============================

class Graph(NeRF.Graph):

    def __init__(self,opt):
        super().__init__(opt)
        self.nerf = NeRF(opt)
        if opt.nerf.fine_sampling:
            self.nerf_fine = NeRF(opt)

    def forward(self,opt,var,mode=None):
        batch_size = len(var.idx)
        pose = self.get_pose(opt,var,mode=mode)
        # render images
        if opt.nerf.rand_rays and mode in ["train","test-optim"]:
            # sample random rays for optimization
            var.ray_idx = torch.randperm(opt.H*opt.W,device=opt.device)[:opt.nerf.rand_rays//batch_size]
            ret = self.render(opt,pose,intr=var.intr,ray_idx=var.ray_idx,mode=mode) # [B,N,3],[B,N,1]
        else:
            # render full image (process in slices)
            ret = self.render_by_slices(opt,pose,intr=var.intr,mode=mode) if opt.nerf.rand_rays else \
                  self.render(opt,pose,intr=var.intr,mode=mode) # [B,HW,3],[B,HW,1]
        var.update(ret)
        return var

    def compute_loss(self,opt,var,mode=None):
        loss = edict()
        batch_size = len(var.idx)
        image = var.image.view(batch_size,3,opt.H*opt.W).permute(0,2,1)
        if opt.nerf.rand_rays and mode in ["train","test-optim"]:
            image = image[:,var.ray_idx]
        # compute image losses
        if opt.loss_weight.render is not None:
            loss.render = self.MSE_loss(var.rgb,image)
        if opt.loss_weight.render_fine is not None:
            assert(opt.nerf.fine_sampling)
            loss.render_fine = self.MSE_loss(var.rgb_fine,image)
        return loss

    def get_pose(self,opt,var,mode=None):
        return var.pose

    def render(self,opt,pose,intr=None,ray_idx=None,mode=None):
        batch_size = len(pose)
        center,ray = camera.get_center_and_ray(opt,pose,intr=intr) # [B,HW,3]
        while ray.isnan().any(): # TODO: weird bug, ray becomes NaN arbitrarily if batch_size>1, not deterministic reproducible
            center,ray = camera.get_center_and_ray(opt,pose,intr=intr) # [B,HW,3]
        if ray_idx is not None:
            # consider only subset of rays
            center,ray = center[:,ray_idx],ray[:,ray_idx]
        if opt.camera.ndc:
            # convert center/ray representations to NDC
            center,ray = camera.convert_NDC(opt,center,ray,intr=intr)
        # render with main MLP
        depth_samples = self.sample_depth(opt,batch_size,num_rays=ray.shape[1]) # [B,HW,N,1]
        rgb_samples,density_samples = self.nerf.forward_samples(opt,center,ray,depth_samples,mode=mode)
        rgb,depth,opacity,prob = self.nerf.composite(opt,ray,rgb_samples,density_samples,depth_samples)
        ret = edict(rgb=rgb,depth=depth,opacity=opacity) # [B,HW,K]
        # render with fine MLP from coarse MLP
        if opt.nerf.fine_sampling:
            with torch.no_grad():
                # resample depth acoording to coarse empirical distribution
                depth_samples_fine = self.sample_depth_from_pdf(opt,pdf=prob[...,0]) # [B,HW,Nf,1]
                depth_samples = torch.cat([depth_samples,depth_samples_fine],dim=2) # [B,HW,N+Nf,1]
                depth_samples = depth_samples.sort(dim=2).values
            rgb_samples,density_samples = self.nerf_fine.forward_samples(opt,center,ray,depth_samples,mode=mode)
            rgb_fine,depth_fine,opacity_fine,_ = self.nerf_fine.composite(opt,ray,rgb_samples,density_samples,depth_samples)
            ret.update(rgb_fine=rgb_fine,depth_fine=depth_fine,opacity_fine=opacity_fine) # [B,HW,K]
        return ret

    def render_by_slices(self,opt,pose,intr=None,mode=None):
        ret_all = edict(rgb=[],depth=[],opacity=[])
        if opt.nerf.fine_sampling:
            ret_all.update(rgb_fine=[],depth_fine=[],opacity_fine=[])
        # render the image by slices for memory considerations
        for c in range(0,opt.H*opt.W,opt.nerf.rand_rays):
            ray_idx = torch.arange(c,min(c+opt.nerf.rand_rays,opt.H*opt.W),device=opt.device)
            ret = self.render(opt,pose,intr=intr,ray_idx=ray_idx,mode=mode) # [B,R,3],[B,R,1]
            for k in ret: ret_all[k].append(ret[k])
        # group all slices of images
        for k in ret_all: ret_all[k] = torch.cat(ret_all[k],dim=1)
        return ret_all

    def sample_depth(self,opt,batch_size,num_rays=None):
        depth_min,depth_max = opt.nerf.depth.range
        num_rays = num_rays or opt.H*opt.W
        rand_samples = torch.rand(batch_size,num_rays,opt.nerf.sample_intvs,1,device=opt.device) if opt.nerf.sample_stratified else 0.5
        rand_samples += torch.arange(opt.nerf.sample_intvs,device=opt.device)[None,None,:,None].float() # [B,HW,N,1]
        depth_samples = rand_samples/opt.nerf.sample_intvs*(depth_max-depth_min)+depth_min # [B,HW,N,1]
        depth_samples = dict(
            metric=depth_samples,
            inverse=1/(depth_samples+1e-8),
        )[opt.nerf.depth.param]
        return depth_samples

    def sample_depth_from_pdf(self,opt,pdf):
        depth_min,depth_max = opt.nerf.depth.range
        # get CDF from PDF (along last dimension)
        cdf = pdf.cumsum(dim=-1) # [B,HW,N]
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]),cdf],dim=-1) # [B,HW,N+1]
        # take uniform samples
        grid = torch.linspace(0,1,opt.nerf.sample_intvs_fine+1,device=opt.device) # [Nf+1]
        unif = 0.5*(grid[:-1]+grid[1:]).repeat(*cdf.shape[:-1],1) # [B,HW,Nf]
        idx = torch.searchsorted(cdf,unif,right=True) # [B,HW,Nf] \in {1...N}
        # inverse transform sampling from CDF
        depth_bin = torch.linspace(depth_min,depth_max,opt.nerf.sample_intvs+1,device=opt.device) # [N+1]
        depth_bin = depth_bin.repeat(*cdf.shape[:-1],1) # [B,HW,N+1]
        depth_low = depth_bin.gather(dim=2,index=(idx-1).clamp(min=0)) # [B,HW,Nf]
        depth_high = depth_bin.gather(dim=2,index=idx.clamp(max=opt.nerf.sample_intvs)) # [B,HW,Nf]
        cdf_low = cdf.gather(dim=2,index=(idx-1).clamp(min=0)) # [B,HW,Nf]
        cdf_high = cdf.gather(dim=2,index=idx.clamp(max=opt.nerf.sample_intvs)) # [B,HW,Nf]
        # linear interpolation
        t = (unif-cdf_low)/(cdf_high-cdf_low+1e-8) # [B,HW,Nf]
        depth_samples = depth_low+t*(depth_high-depth_low) # [B,HW,Nf]
        return depth_samples[...,None] # [B,HW,Nf,1]


class VolNeRF(NeRF.NeRF):

    def __init__(self,opt):
        super().__init__()
        self.define_network(opt)

    def define_network(self,opt):
        
        
        
        input_3D_dim = 3+6*opt.arch.posenc.L_3D if opt.arch.posenc else 3
        if opt.nerf.view_dep:
            input_view_dim = 3+6*opt.arch.posenc.L_view if opt.arch.posenc else 3
        # point-wise feature and output density
        if "use_tcnn" in opt.arch and opt.arch.use_tcnn:
        # Using TCNN"""
            self.mlp_feat = tcnn.Network(n_input_dims=input_3D_dim,
                                         n_output_dims=opt.arch.feature_network_config.n_neurons+1,
                                         network_config=opt.arch.feature_network_config)
        elif "use_tcnn" in opt.arch and not opt.arch.use_tcnn:
        # Using normal pytorch
            self.mlp_feat = torch.nn.ModuleList()
            L = util.get_layer_dims(opt.arch.layers_feat)
            for li,(k_in,k_out) in enumerate(L):
                if li==0: k_in = input_3D_dim
                if li in opt.arch.skip: k_in += input_3D_dim
                if li==len(L)-1: k_out += 1
                linear = torch.nn.Linear(k_in,k_out)
                if opt.arch.tf_init:
                    self.tensorflow_init_weights(opt,linear,out="first" if li==len(L)-1 else None)
                self.mlp_feat.append(linear)
        # RGB prediction
        if "use_tcnn" in opt.arch and opt.arch.use_tcnn:
            self.mlp_rgb = tcnn.Network(n_input_dims=opt.arch.feature_network_config.n_neurons,
                                        n_output_dims=3,
                                        network_config=opt.arch.rgb_network_config)
        elif "use_tcnn" in opt.arch and not opt.arch.use_tcnn:
            self.mlp_rgb = torch.nn.ModuleList()
            L = util.get_layer_dims(opt.arch.layers_rgb)
            feat_dim = opt.arch.layers_feat[-1]
            for li,(k_in,k_out) in enumerate(L):
                if li==0: k_in = feat_dim+(input_view_dim if opt.nerf.view_dep else 0)
                linear = torch.nn.Linear(k_in,k_out)
                if opt.arch.tf_init:
                    self.tensorflow_init_weights(opt,linear,out="all" if li==len(L)-1 else None)
                self.mlp_rgb.append(linear)

    def positional_encoding(self,opt,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device=opt.device)*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc
