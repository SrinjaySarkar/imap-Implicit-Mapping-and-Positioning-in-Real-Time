import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import copy


class imap_model(torch.nn.Module):
    def __init__(self):
        super(imap_model,self).__init__()
        self.pos_enc=torch.nn.Linear(3,50,bias=False).cuda()
        torch.nn.init.normal_(self.pos_enc.weight,0.0,25.0)
        self.fc1=torch.nn.Linear(50,128).cuda()
        self.fc2=torch.nn.Linear(128,128).cuda()
        self.fc3=torch.nn.Linear(128+50,128).cuda()
        self.fc4=torch.nn.Linear(128,128).cuda()
        self.fc5=torch.nn.Linear(128,4,bias=False).cuda()
        self.fc5.weight.data[3,:]*=0.1

    def forward(self,p):
        gamma=torch.sin(self.pos_enc(p))
        # print("g:",gamma.shape)
        o1=F.relu(self.fc1(gamma))
        # print("o1:",o1.shape)
        # print(self.fc2(o1).shape)
        o2=torch.cat([F.relu(self.fc2(o1)),gamma],dim=1)
        # print(self.fc2(o1).shape)
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
