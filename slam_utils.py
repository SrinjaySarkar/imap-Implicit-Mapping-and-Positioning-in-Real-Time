import numpy as np
import torch
import cv2
import os
import csv
from model import camera,imap_model


# model=imap_model().cuda()
# tracking_model=model.cuda()
# tracking_model.load_state_dict(model.state_dict())

#coarsee_points,weights,n_fine
def sampling(bins,weights,nf): # try naive hierarchical sampling
    weights+=1e-5#avoid underflow
    # print(weights.shape)
    # print(torch.sum(weights,dim=-1,keepdim=True).shape)
    pdf=weights/(torch.sum(weights,-1,keepdim=True))
    # print(pdf.shape)
    cdf=torch.cumsum(pdf,-1)#get cdf by integrating pdf
    # print(cdf.shape)
    cdf=torch.cat([torch.zeros_like(cdf[...,:-1]),cdf],-1)
    #inverse transform sampling
    u=torch.rand(list(cdf.shape[:-1])+[nf]).cuda()#unformly sampled
    # print(u.shape)
    u=u.contiguous()
    idxs=torch.searchsorted(cdf,u,right=True)
    low=torch.max(torch.zeros_like(idxs-1),idxs-1)
    high=torch.min((cdf.shape[-1]-1)*torch.ones_like(idxs),idxs)
    idxs_g=torch.stack([low,high],-1)

    matched_shape=[idxs_g.shape[0],idxs_g.shape[1],cdf.shape[-1]]
    cdf_g=torch.gather(cdf.unsqueeze(1).expand(matched_shape),2,idxs_g)
    bins_g=torch.gather(cdf.unsqueeze(1).expand(matched_shape),2,idxs_g)

    denom=(cdf_g[...,1]-cdf_g[...,0])
    denom=torch.where(denom<1e-5,torch.ones_like(denom),denom)
    t=(u-cdf_g[...,0])/denom
    samples=bins_g[...,0]+t*(bins_g[...,1]-bins_g[...,0])
    samples_cat,_=torch.sort(torch.cat([samples,bins],-1),dim=-1)
    return(samples_cat)



def rays(camera,u,v):
    """ray_origins:denotes the origin of the ray passing through pixel[u,v] shape:(bs,3)
       ray_directions:denotes the direction of the ray passing through the pixel[u,v] shape:(bs,3)"""
    pixel=torch.cuda.FloatTensor(v.shape[0],3,1).fill_(1)
    # print(pixel.shape)
    pixel[:,0,0]=u
    pixel[:,1,0]=v
    #homogenized pixel coordinate system
    a=torch.matmul(camera.K_inv,pixel)
    world_coord=torch.matmul(camera.R,torch.matmul(camera.K_inv,pixel))#[u.shape[0],3,1]
    world_coord=world_coord[:,:,0]#[u.shape[0],3]
    # print(world_coord[np.random.randint(0,u.shape[0])])
    norm_world_coord=torch.norm(world_coord,dim=1)#normalized ray direction
    # print(world_coord.shape)
    norm_world_coord=norm_world_coord.reshape(world_coord.shape[0],1).expand(world_coord.shape[0],3)
    with torch.no_grad():
        world_coord/=norm_world_coord
    # print(world_coord.shape)
    # print(world_coord[0])
    # print(torch.norm(world_coord[0]))
    ray_origin=camera.T.reshape(1,camera.T.shape[0]).expand(u.shape[0],camera.T.shape[0])
    # print(ray_origin.shape)
    ray_dir=world_coord
    # print(ray_dir,ray_origin)
    return (ray_dir,ray_origin)#ray origin is the translation vector and the ray direction is the world coordinate.


def render_volume(points,model_op):
    max_dists=1.5
    step=points.shape[1]
    #slow
    # p=[]
    # for i in range(points.shape[1]-1):
    #     p.append(points[:,i+1]-points[:,i])
    # print(len(p))

    dists=points[:,1:]-points[:,:-1]#distance between consecutive points since the fine points are not sampled uniformly
    #hence different distance between two pairs of points
    # o=1-torch.exp(-coarse_model_op[:,:,3]*delta)
    # # print(o.shape)
    # o=o[:,1:]
    # # print(o.shape)
    # t=1-torch.exp(-torch.cumsum(coarse_model_op[:,:,3]*delta,1))[:,:-1]#sub value of o in equation; and (j=1 to i-1) so remove last one.
    # """try this also: t=torch.exp(-torch.cumsum(model_op[:,:,3]*delta,1))[:,:-1] """
    # # print(t.shape)
    # w=o*t

    o=1-torch.exp(-model_op[:,:-1,3]*dists)#dist of d_i to d_i+1 does not exist
    # print(o.shape)
    wo=torch.cumprod((1-o),dim=1)
    ws=torch.cuda.FloatTensor(points.shape[0],points.shape[1]-1).fill_(0)#first element of ws will always be 0 since j=1 to i-1(see paper eq1).
    ws[:,1:]=o[:,1:]*wo[:,:-1]
    d_cap=ws*points[:,:-1]
    d=torch.sum(d_cap,dim=1)
    i_cap=ws.unsqueeze(2).expand(points.shape[0],-1,3)*model_op[:,:-1,:3]
    i=torch.sum(i_cap,dim=1)
    d_var=torch.sum(ws*torch.square(points[:,:-1]-d.reshape(points.shape[0],1).expand(points.shape[0],step-1)),dim=1)#check this again?
    return (d,i,d_var)


def render_pixel_rays(u,v,camera,imap_model,tracking_model,nc,nf,track=False):
    if track:
        model=tracking_model
    else:
        model=imap_model
    #we have pixels , we need the fine and coarse points along with the weights.
    batch_size=u.shape[0]
    ray_dir,ray_origin=rays(camera,u,v)
    # print(ray_dir.shape)
    # print(ray_origin.shape)
    with torch.no_grad():
        coarse_points=torch.linspace(0.0001,1.2,nc).cuda().reshape(1,nc).expand(u.shape[0],nc)#sample points on ray and then repeat for all pixels.
        cp1=coarse_points
        # print(coarse_points.shape)
        #p=o+r*di ; p: set of points on the ray through the pixel ; o : ray origin all have same origin since all pixels belong to same image;
        #r is the ray direction which is different for every pixel; d is the distance between the sampled points since we have the points directly we multiply them
        rays1=ray_origin.unsqueeze(1).expand(u.shape[0],nc,3) #+ ray_dir*points
        # print(rays1.shape)
        ray_dir1=ray_dir.unsqueeze(1).expand(u.shape[0],nc,3)
        coarse_points=coarse_points.unsqueeze(2).expand(u.shape[0],nc,3)
        # print("o:",rays1.shape)
        # print("r:",ray_dir.shape)
        # print("d:",coarse_points.shape)

        coarse_rays=rays1 + (ray_dir1*coarse_points)#o+r*di
        #after checking the points are equidistant from one another but are collinear to an delta of 1e-3. so adding noise mgiht be a good idea?
        coarse_model_op=model(coarse_rays.reshape(-1,3))#(bs*nc,4)(r,g,b,rho)
        # print(model_op.shape)
        coarse_model_op=coarse_model_op.reshape(u.shape[0],nc,4)
        delta=cp1[0,1]-cp1[0,0]

        # print(coarse_model_op[:,:,3].shape)
        o=1-torch.exp(-coarse_model_op[:,:,3]*delta)
        # print(o.shape)
        o=o[:,1:]
        # print(o.shape)
        t=1-torch.exp(-torch.cumsum(coarse_model_op[:,:,3]*delta,1))[:,:-1]#sub value of o in equation; and (j=1 to i-1) so remove last one.
        """try this also: t=torch.exp(-torch.cumsum(model_op[:,:,3]*delta,1))[:,:-1] """
        # print(t.shape)
        w=o*t

        # print(w.shape)
        fine_points=sampling(cp1,w,nf)#(bc+nf)
        # print(fine_points.shape)

    rays2=ray_origin.unsqueeze(1).expand(u.shape[0],nc+nf,3)
    ray_dir2=ray_dir.unsqueeze(1).expand(u.shape[0],nc+nf,3)
    fine_points1=fine_points.unsqueeze(2).expand(u.shape[0],nc+nf,3)

    fine_rays=rays2+ray_dir2*fine_points1
    fine_model_op=model(fine_rays.reshape(-1,3))
    fine_model_op=fine_model_op.reshape(u.shape[0],nc+nf,4)
    # print(fine_model_op.shape)
    d_image,c_image,d_var=render_volume(fine_points,fine_model_op)
    # print(d_image.shape)
    # print(c_image.shape)
    # print(d_var.shape)
    c_image=camera.exp_a*c_image+camera.params[7]
    return (d_image,c_image,d_var)



def render(camera,label,model,tracking_model,idx):
    with torch.no_grad():
        camera.update_transform()
        height=int(camera.size[0]/5)
        width=int(camera.size[1]/5)
        rgb=torch.cuda.FloatTensor(height,width,3).fill_(0)
        depth=torch.cuda.FloatTensor(height,width).fill_(0)
        vs=5*torch.arange(height).reshape(height,1).expand(height,width).reshape(-1).cuda()
        us=5*torch.arange(width).reshape(1,width).expand(height,width).reshape(-1).cuda()
        # d_f,i_f,dv_f=render_pixel_rays(us,vs,camera,track=False)

        d_f,i_f,dv_f=render_pixel_rays(us,vs,camera,model,tracking_model,track=False,nc=32,nf=12)

        depth=d_f.reshape(-1,width)
        rgb=i_f.reshape(-1,width,3)
        rgb_cv=torch.clamp(rgb*255,0,255).detach().cpu().numpy().astype(np.uint8)
        depth_cv=torch.clamp(depth*50000/256,0,255).detach().cpu().numpy().astype(np.uint8)

        rgb_gt=torch.clamp(camera.rgb_images*255,0,255).detach().cpu().numpy().astype(np.uint8)
        depth_gt=torch.clamp(camera.depth_images*50000/256,0,255).detach().cpu().numpy().astype(np.uint8)
        prev_rgb=cv2.hconcat([cv2.resize(rgb_cv, (camera.size[1], camera.size[0])), rgb_gt])
        prev_depth=cv2.cvtColor(cv2.hconcat([cv2.resize(depth_cv, (camera.size[1], camera.size[0])), depth_gt]), cv2.COLOR_GRAY2RGB)
        prev=cv2.vconcat([prev_rgb, prev_depth])
        cv2.imwrite("/content/render{}_{:04}.png".format(label,idx),prev)
        print("image saved")
        idx+=1
        print("idx:",idx)
