import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import copy

def mish(x):
    res=x*torch.tanh(F.softplus(x))
    return (res)

class imap_model(torch.nn.Module):
    def __init__(self):
        super(imap_model,self).__init__()
        self.pos_enc=torch.nn.Linear(3,93,bias=False)
        torch.nn.init.normal_(self.pos_enc.weight,0.0,25.0)
        self.fc1=torch.nn.Linear(93,256)
        self.fc2=torch.nn.Linear(256,256)
        self.fc3=torch.nn.Linear(93+256,256)
        self.fc4=torch.nn.Linear(256,256)
        self.fc5=torch.nn.Linear(256,4,bias=False)
        self.fc5.weight.data[3,:]*=0.1
    
    def forward(self,p):
        gamma=torch.sin(self.pos_enc(p))
        o1=F.relu(self.fc1(gamma))
        o2=torch.cat([F.relu(self.fc2(o1)),gamma],dim=1)
        o3=F.relu(self.fc3(o2))
        o4=F.relu(self.fc4(o3))
        out=self.fc5(o4)
        return (out)


class camera():
    def __init__(self,rgb,depth,px,py,pz,rx,ry,rz,a=0.0,b=0.0,fx=525.0,fy=525.0,cx=319.5,cy=239.5):
        self.params=torch.tensor([rx,ry,rz,px,py,pz,a,b]).detach().cuda().requires_grad_(True)
        #calibrating camera
        self.fx=fx
        self.fy=fy
        self.cx=cx
        self.cy=copy
        self.K=torch.from_numpy(np.array([[fx,0.0,cx],[0.0,fy,cy],[0.0,0.0,1.0]]).astype(np.float32)).cuda().requires_grad_(True)
        self.K_inv=torch.from_numpy(np.array([[1.0/fx,0.0,-cx/fx],[0.0,1.0/fy,-cy/fy],[0.0,0.0,1.0]]).astype(np.float32)).cuda().requires_grad_(False)
        self.rgb_images=torch.from_numpy((rgb).astype(np.float32)).cuda()
        self.rgb_images/=256 #normalize
        self.depth_images=torch.from_numpy((depth).astype(np.float32)).cuda()
        self.depth_images/=50000 #depth to 16 bit color

        self.exp_a=torch.FloatTensor(1).cuda()
        self.R=torch.zeros(3,3).cuda()
        self.T=torch.zeros(3,3).cuda()
        self.Li=torch.cuda.FloatTensor(64).fill_(1.0/64)
        ## normalized loss on the pixels common in both the 8*8 patch and the sampled set ; see equation 7 in papre 
        self.size=depth.shape

        #calculating the pose matrix of the camera 
        self.update_transform()
        self.optimizer=torch.optim.Adam([self.params],lr=0.005)
    
    def set_images(self,rgb,depth):
        self.rgb_images=torch.from_numpy((rgb).astype(np.float32)).cuda()
        self.rgb_images/=256 #normalize
        self.depth_images=torch.from_numpy((depth).astype(np.float32)).cuda()
        self.depth_images/=50000 #depth to 16 bit color

    # Calc Transform from camera parameters
    def update_transform(self): # do this better , doesn't make sense now but works 
        i = torch.cuda.FloatTensor(3,3).fill_(0)
        i[0,0] = 1
        i[1,1] = 1
        i[2,2] = 1
        w1 = torch.cuda.FloatTensor(3,3).fill_(0)
        w1[1, 2] = -1
        w1[2, 1] = 1
        w2 = torch.cuda.FloatTensor(3,3).fill_(0)
        w2[2, 0] = -1
        w2[0, 2] = 1
        w3 = torch.cuda.FloatTensor(3,3).fill_(0)
        w3[0, 1] = -1
        w3[1, 0] = 1

        #getting the pose matrix because self.R is th pose matrix check this logic and other logic if they have the same output   
        th = torch.norm(self.params[0:3])
        thi = 1.0/(th+1e-12)
        n = thi * self.params[0:3]
        c = torch.cos(th)
        s = torch.sin(th)
        w = n[0]*w1 + n[1]*w2 + n[2]*w3
        ww = torch.matmul(w,w)
        R1 = i + s * w 
        self.R = R1 + (1.0-c)*ww
        self.T = self.params[3:6]
        self.exp_a = torch.exp(self.params[6])
    
    def rays(self,u,v): # read the paragraph 1 of section 3.3 if any doubts
        # u,v are the pixels in the rendered image
        pixel=torch.cuda.FloatTensor(u.shape[0],3,1).fill_(1)
        #turn pixel into homogenous coordinate for all pixels in the batch
        pixel[:,0,0]=u
        pixel[:,1,0]=v
        a=torch.matmul(self.K_inv,pixel)
        world_coord=torch.matmul(self.R,torch.matmul(self.K_inv,pixel))[:,:,0] # corrsponding 3d point in the world for every [u,v] pixel
        with torch.no_grad():
            world_coord/=torch.norm(world_coord,dim=1).reshape(u.shape[0],1).expand(u.shape[0],3)#normalized world coord
        ray_origin=self.T.reshape(1,3).expand(u.shape[0],3)
        ray_dir=world_coord
        return (ray_dir,ray_origin)
    
def hierarchical_sampling(bins,weights,N_samples):
    #read section 5.3 in Nerf but chnage weights accoridng to SLAM
    weights+=1e-5
    piecewise_pdf=weights/(torch.sum(weights,-1,keepdim=True))
    #get cdf from pdf
    cdf=torch.cumsum(piecewise_pdf,-1)
    cdf=torch.cat([torch.zeros_like(cdf[...,:-1]),cdf],-1)# (bs,bins)
    #inverse transform sampling
    u=torch.rand(list(cdf.shape[:-1]) + [N_samples])#unifrom samples from bins
    u=u.contiguous()
    idxs=torch.searchsorted(cdf,u,right=True)
    low=torch.max(torch.zeros_like(idxs-1),idxs-1)
    high=torch.min((cdf.shape[-1]-1)*torch.ones_like(idxs),idxs)
    idxs_g=torch.stack([low,high],-1)

    matched_shape=[idxs_g.shape[0],idxs_g.shape[1],cdf.shape[-1]]
    cdf_g=torch.gather(cdf.unsqueeze(1).expand(macthed_shape),2,idxs_g)
    bins_g=torch.gather(cdf.unsqueeze(1).expand(macthed_shape),2,idxs_g)
    denom=(cdf_g[...,1]-cdf_g[...,0])
    denom=torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t=(u-cdf_g[...,0])/denom
    samples=bins_g[...,0]+t*(bins_g[...,1]-bins_g[...,0])
    samples_cat,_=torch.sort(torch.cat([samples, bins], -1), dim=-1)
    return (samples_cat)


class imap_SLAM():
    def __init__(self):
        self.i=0
        self.model=imap_model().cuda()
        self.tracking_model=imap_model().cuda()
        self.cameras=[]
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=0.005)
    
    def opt_tracking_model(self):
        self.tracking_model.load_state_dict(self.model.state_dict())
    
    def add_camera(self,rgb,depth,px,py,pz,rx,ry,rz,a,b):
        flag_rgb=cv2.IMREAD_COLOR
        rgb_image=cv2.imread(rgb,flag_rgb)
        flag_depth=cv2.IMREAD_ANYDEPTH
        depth_image=cv2.imread(depth,flag_depth)
        cam=camera(rgb_image,depth_image,px,py,pz,rx,ry,rz,a,b)
        self.cameras.append(cam)
    
    def render_volume(self,dists,sigmas):
        #dists:(bs,n_coarse+n_fine,3)
        #sigmas:(batch_size,n_coarse+n_fine, 4)4 because complete model output rgb+density 
        max_dist=1.5
        batch_size=dists.shape[0]
        step=dists.shape[1]#n points so n steps
        deltas=dists[:,1:]-dists[:,:-1]#distance between points (sum of all excpet last)-(sum of all except last)
        o=1-torch.exp(-sigmas[:,:-1,3]*deltas)#all the denstities except last since the product is from 1 to i-1
        wo=torch.cumprod((1-o),1)
        w=torch.FloatTensor(batch_size,step-1).fill_(0)
        w[:,1:]=o[:,1:]*wo[:,:-1]#ll the denstities except last since the product is from 1 to i-1
        d_cap=w*dists[:,:-1]#eq1 in paper
        i_cap=w.reshape(batch_size,-1,1).expand(batch_size-1,3)*sigmas[:,:-1,:3]#eq1 of paper
        d_var=torch.sum(w*torch.square(dists[:, :-1]-d.reshape(batch_size,1).expand(batch_size,step-1)), dim=1)
        d=torch.sum(d_cap,dim=1)
        i=torch.sum(i_cap,dim=1)
        # d+=wo[:,-1]*max_dist
        return (d,i,d_var)
    
    def render_pixel_rays(self,u,v,camera,nc=32,nf=12,track=False):
        if track:
            model=self.tracking_model
        else:
            model=self.model
        batch_size=u.shape[0]
        ray,origin=camera.rays(u,v)

        with torch.no_grad():
            ds=torch.linspace(0.0001,1.2,nc).cuda().reshape(1,nc).expand(batch_size,nc)
            #p=o+r*di
            rays=origin.reshape(batch_size,1,3).expand(batch_size,nc,3)+ray.reshape(batch_size,1,3).expand(batch_size,nc,3)*ds.reshape(batch_size,nc,1).expand(batch_size,nc,3)
            # print(rays.shape)
            # print(model)
            sigmas=model(rays.reshape(-1,3))
            sigmas=sigmas.reshape(batch_size,nc,4)
            delta=ds[0,2]-ds[0,1]
            o=1-torch.exp(-sigmas[:, :,3]*delta)[:,1:]
            t=1-torch.exp(-torch.cumsum(sigmas[:, :,3]*delta,1))[:,:-1]#???
            w=o*t
            ds_fine=hierarchical_sampling(ds,w,nf)
        #p=o+r*di
        rays_f=origin.reshape(batch_size,1,3).expand(batch_size,nc+nf,3)+ray.reshape(batch_size,1,3).expand(batch_size,nc+nf,3)* ds_fine.reshape(batch_size,nc+nf,1).expand(batch_size,nc+nf,3)
        sigmas_f=model(rays_f.reshape(-1,3)).reshape(batch_size,nc+nf,4)
        d_f,i_f,dv_f=self.render_volume(ds_fine,sigmas_f)
        i_f=cam.exp_a*i_f+cam.params[7]
        return (d_f,i_f,dv_f)
    
    def render(self,camera,label):
        with torch.no_grad():
            camera.update_transform()
            height=int(camera.size[0]/2)
            width=int(camera.size[1]/2)
            rgb=torch.cuda.FloatTensor(height,width,3).fill_(0)
            depth=torch.cuda.FloatTensor(height,width).fill_(0)
            vs=2*torch.arange(height).reshape(height,1).expand(height,width).reshape(-1).cuda()
            us=2*torch.arange(width).reshape(1,width).expand(height,width).reshape(-1).cuda()
            d_f,i_f,dv_f=self.render_pixel_rays(us,vs,camera,track=False)

            depth=d_f.reshape(-1,w)
            rgb=i_f.reshape(-1,w,3)
            rgb_cv=torch.clamp(rgb*255,0,255).detach().cpu().numpy().astype(np.uint8)
            depth_cv=torch.clamp(depth*50000/256,0,255).detach().cpu().numpy().astype(np.uint8)

            rgb_gt=torch.clamp(camera.rgb*255,0,255).detach().cpu().numpy().astype(np.uint8)
            depth_gt=torch.clamp(camera.depth*50000/256,0,255).detach().cpu().numpy().astype(np.uint8)
            prev_rgb=cv2.hconcat([cv2.resize(rgb_cv, (camera.size[1], camera.size[0])), rgb_gt])
            prev_depth=cv2.cvtColor(cv2.hconcat([cv2.resize(depth_cv, (camera.size[1], camera.size[0])), depth_gt]), cv2.COLOR_GRAY2RGB)
            prev=cv2.vconcat([prev_rgb, prev_depth])
            cv2.imwrite("render/{}_{:04}.png".format(label,self.i),prev)
            self.i+=1
    
    def map(self,batch_size=200,active_sampling=True):
        if len(self.cameras)<5:
            camera_ids=np.arange(len(self.cameras))
        else:
            camera_ids=np.random.randint(0,len(self.cameras)-2,5)
            camera_ids[3]=len(self.cameras)-1
            camera_ids[4]=len(self.cameras)-2
        
        for camera_id in camera_ids:
            # print(camera_id)
            # print(self.cameras[camera_id].Li[0])
            self.optimizer.zero_grad()
            self.cameras[camera_id].optimizer.zero_grad()
            self.cameras[camera_id].update_transform()

            height=self.cameras[camera_id].size[0]
            width=self.cameras[camera_id].size[1]
            if active_sampling: # sample random pixels from frame 
                with torch.no_grad():
                    sh=int(height/8)
                    sw=int(width/8)
                    ul,vl=[],[]
                    ri=torch.cuda.FloatTensor(64).fill_(0)
                    # print(self.cameras[camera_id])
                    # print()
                    for i in range(64):
                        # print(self.cameras[camera_id].Li[i])
                        ni=int(batch_size*self.cameras[camera_id].Li[i])
                        #print(ni)
                        if ni<1:
                            ni=1
                        ri[i]=ni
                        ul.append((torch.rand(ni)*(sw-1)).to(torch.int16).cuda() + (i%8)*sw)
                        vl.append((torch.rand(ni)*(sh-1)).to(torch.int16).cuda()+ int(i/8)*sh)
                    us=torch.cat(ul)
                    vs=torch.cat(vl)
            else:
                #take all pixels from frame
                us=((torch.rand(batch_size)*(width-1))).to(torch.int16).cuda()
                vs=((torch.rand(batch_size)*(height-1))).to(torch.int16).cuda()
            
            depth,rgb,depth_variance=self.render_pixel_rays(us,vs,self.cameras[camera_id],track=False)
            rgb_gt=torch.cat([self.cameras[camera_id].rgb_images[v,u,:].unsuqeeze(0)for u,v in zip(us,vs)])
            depth_gt=torch.cat([self.cameras[camera_id].depth_images[v,u].unsuqeeze(0)for u,v in zip(us,vs)])
            depth[depth_gt==0]=0 #so that they don't affect the loss value
            with torch.no_grad():
                var=torch.reciprocal(torch.sqrt(depth_var))
                var[var.isinf()]=1
                var[var.isnan()]=1
            g_loss=torch.mean(torch.abs(depth-depth_gt)*var)
            p_loss=torch.mean(torch.abs(rgb-rgb_gt))

            total_loss=(g_loss) + (5*p_loss)
            total_loss.backward()
            self.optimizer.step() 

            if camera_id>0:
                self.cameras[camera_id].optimizer.step()
            if active_sampling: # read eq 7 and eq 8 
                with torch.no_grad():
                    e=torch.abs(depth-depth_gt)+torch.sum(torch.abs(rgb-rgb_gt),1)
                    rs=torch.cumsum(ri,0)
                    Li=torch.zeros(64).type(torch.IntTensor)
                    Li[0]=torch.mean(e[:ris[0]])
                    for i in range(1,64):
                        Li[i]=torch.mean(e[ris[i-1]:ris[i]])
                    d=1.0/torch.sum(Li)
                    self.cameras[camera_id].Li=d*Li
    
    def track(self,camera,batch_size=200):
        self.opt_tracking_model()
        for i in range(20):
            camera.optimizer.zero_grad(20)
            camera.update_transform()
            height=camera.size[0]
            width=camera.size[1]
            us=(torch.rand(batch_size)*(w-1)).to(torch.int16).cuda()
            vs=(torch.rand(batch_size)*(h-1)).to(torch.int16).cuda()
            depth,rgb,depth_variance=self.render_pixel_rays(us,vs,camera,track=True)
            depth_gt=torch.cat([camera.depth_images[v,u].unsqueeze(0) for u,v in zip(us,vs)])
            rgb_gt=torch.cat([camera.rgb_images[v,u,:].unsqueeze(0) for u,v in zip(us,vs)])
            depth[depth_gt==0]=0 
            with torch.no_grad():
                var=torch.reciprocal(torch.sqrt(depth_var))
                var[var.isinf()]=1
                var[var.isnan()]=1
            g_loss=torch.mean(torch.abs(depth-depth_gt)*var)
            p_loss=torch.mean(torch.abs(rgb-rgb_gt))

            total_loss=(g_loss) + (5*p_loss)
            total_loss.backward()
            camera.optimizer.step()
            p=float(torch.sum(((torch.abs(depth-depth_gt)*torch.reciprocal(depth_gt+1e-12))<0.1).int()).cpu().item()) /batch_size
            if p>0.80: # see equation 6 for key frmae
                break
        return (p)


    
