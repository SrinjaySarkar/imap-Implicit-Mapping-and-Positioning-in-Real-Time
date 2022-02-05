import numpy as np
import torch
import cv2
import os
import csv
from slam_utils import sampling,rays,render_volume,render_pixel_rays,render
from model import imap_model,camera


class imap_slam():
    def __init__(self):
        self.model=imap_model().cuda()
        self.tracking_model=imap_model().cuda()
        self.cameras=[]
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=0.005)
    
    def init_tracking_model(self):
        self.tracking_model.load_state_dict(self.model.state_dict())
    
    def add_camera(self,rgb,depth,px,py,pz,rx,ry,rz,a,b):
        flag_rgb=cv2.IMREAD_COLOR
        rgb_image=cv2.imread(rgb,flag_rgb)
        flag_depth=cv2.IMREAD_ANYDEPTH
        depth_image=cv2.imread(depth,flag_depth)
        cam=camera(rgb_image,depth_image,px,py,pz,rx,ry,rz,a,b)
        self.cameras.append(cam)
        # print(len(self.cameras))
    
    def map(self,batch_size=200,active_sampling=True):
        # print(len(cameras))
        if len(self.cameras)<5:
            camera_ids=np.arange(len(self.cameras))
        else:
            camera_ids=np.random.randint(0,len(self.cameras)-2,5)#maybe because we only window_size=5.see table 4 
            camera_ids[3]=len(self.cameras)-1
            camera_ids[4]=len(self.cameras)-2
        # print("Camera ids:",camera_ids)
        ####
        for camera_id in camera_ids:
            # print(self.cameras[camera_id])
            self.optimizer.zero_grad()
            self.cameras[camera_id].optimizer.zero_grad()
            self.cameras[camera_id].update_transform()
            # print("pose matrix:",self.cameras[camera_id].R) # could rotation matrix be used as a pose matrix ???
            # print("translation matrix:",self.cameras[camera_id].T)# translation matrix

            height=self.cameras[camera_id].size[0]
            width=self.cameras[camera_id].size[1]
            # print("Image height:",height)
            # print("Image width:",width)

            if active_sampling:
                with torch.no_grad():
                    sh=int(height/8)#dividing the image into a 8*8 grid so grid is 64 units
                    sw=int(width/8)
                    # print(sw,sh)
                    ul,vl=[],[]
                    ri=torch.cuda.IntTensor(64).fill_(0)
                    for i in range(64):
                        ni=int(batch_size*self.cameras[camera_id].Li[i])#200*loss_value over that cell of an (8*8) grid
                        #the way this is done is not exactly an intersection;multiply 200 * (Li/sum(Li) which gives a n_weight, then sample n_weight 
                        #random pixel locations from each patch)
                        if ni<1:
                            ni=1
                        ri[i]=ni#ri contains how many pixels were sampled for each unit in the grid
                        #sample pixel locations for each grid no of pixels in each grid location is ni or ri[i] 
                        ul.append((torch.rand(ni)*(sw-1)).to(torch.int16).cuda() + (i%8)*sw)
                        vl.append((torch.rand(ni)*(sh-1)).to(torch.int16).cuda() + int(i/8)*sh)
                    us=torch.cat(ul)
                    vs=torch.cat(vl)
                    
            else:
                #take all pixels
                us=((torch.rand(batch_size)*(width-1))).to(torch.int16).cuda()
                vs=((torch.rand(batch_size)*(height-1))).to(torch.int16).cuda()
            depth,rgb,depth_variance=render_pixel_rays(us,vs,self.cameras[camera_id],self.model,self.tracking_model,nc=32,nf=12,track=False)
            rgb_gt=torch.cat([self.cameras[camera_id].rgb_images[v,u,:].unsqueeze(0) for u,v in zip(us,vs)])
            depth_gt=torch.cat([self.cameras[camera_id].depth_images[v,u].unsqueeze(0) for u,v in zip(us,vs)])
            depth[depth_gt==0]=0
            with torch.no_grad():
                var=torch.reciprocal(torch.sqrt(depth_variance))
                var[var.isinf()]=1
                var[var.isnan()]=1
            g_loss=torch.mean(torch.abs(depth-depth_gt)*var)
            p_loss=torch.mean(torch.abs(rgb-rgb_gt))

            total_loss=(g_loss)+(5*p_loss)
            # print(total_loss)
            total_loss.backward()
            self.optimizer.step()
            #look into the folowing snippet
            if camera_id>0:
                self.cameras[camera_id].optimizer.step()
            if active_sampling:
                with torch.no_grad():
                    e=torch.abs(depth-depth_gt)+torch.sum(torch.abs(rgb-rgb_gt),1)
                    ris=torch.cumsum(ri,0)
                    Li=torch.cuda.FloatTensor(64).fill_(0)
                    Li[0]=torch.mean(e[:ris[0]])
                    for i in range(1,64):
                        Li[i]=torch.mean(e[ris[i-1]:ris[i]])
                    d=1.0/torch.sum(Li)
                    self.cameras[camera_id].Li=d*Li
    
    def track(self,camera,batch_size=200):
        tracking_model=self.init_tracking_model()
        for _ in range(20):
            camera.optimizer.zero_grad()
            camera.update_transform()
            height=camera.size[0]
            width=camera.size[1]
            us=(torch.rand(batch_size)*(width-1)).to(torch.int16).cuda()
            vs=(torch.rand(batch_size)*(height-1)).to(torch.int16).cuda()
            depth,rgb,depth_variance=render_pixel_rays(us,vs,camera,self.model,self.tracking_model,nc=32,nf=12,track=True)
            rgb_gt=torch.cat([camera.rgb_images[v,u,:].unsqueeze(0) for u,v in zip(us,vs)])
            depth_gt=torch.cat([camera.depth_images[v,u].unsqueeze(0) for u,v in zip(us,vs)])
            depth[depth_gt==0]=0
            with torch.no_grad():
                var=torch.reciprocal(torch.sqrt(depth_variance))
                var[var.isinf()]=1
                var[var.isnan()]=1
            g_loss=torch.mean(torch.abs(depth-depth_gt)*var)
            p_loss=torch.mean(torch.abs(rgb-rgb_gt))

            total_loss=(g_loss) + (5*p_loss)
            total_loss.backward()
            camera.optimizer.step()
            p=float(torch.sum(((torch.abs(depth-depth_gt)*torch.reciprocal(depth_gt+1e-12))<0.1).int()).cpu().item()) /batch_size
            if p>0.80:
                break
        print ("Tracking :",p)
        return (p)
