import torch
import sys
import glob
import os 
import csv
import cv2, threading 
from imap_all_model import camera,imap_SLAM


data_path="/vinai/sskar/iMAP/rgbd_dataset_freiburg1_teddy/"
def read_files(root_path):
    rgb_files,depth_files=[],[]
    rgb_csv_file=open(root_path+"rgb.txt","r")
    f=csv.reader(rgb_csv_file,delimiter=" ")
    for _ in range(3):
        next(f)
    for row in f:
        rgb_files.append(os.path.join(root_path,row[1]))
    
    depth_csv_file=open(root_path+"depth.txt","r")
    f=csv.reader(depth_csv_file,delimiter=" ")
    for _ in range(3):
        next(f)
    for row in f:
        depth_files.append(os.path.join(root_path,row[1]))
    
    return (rgb_files,depth_files)

def mapping_thread(imap_SLAM):
    while True:
        imap_SLAM.map()
        time.sleep(0.1)
        print("updated map")

def main():
    mapper=imap_SLAM()
    rgb_files,depth_files=read_files(data_path)
    frame_length=min(len(rgb_files),len(depth_files))
    #init camera
    mapper.add_camera(rgb_files[0],depth_files[0],0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    fixed_camera=camera(cv2.imread(rgb_files[0],cv2.IMREAD_COLOR),cv2.imread(depth_files[0], cv2.IMREAD_ANYDEPTH),0.0,0.0,0.0,1e-8,1e-8,1e-8,0.0,0.0)
    tracking_camera=camera(cv2.imread(rgb_files[0],cv2.IMREAD_COLOR),cv2.imread(depth_files[0], cv2.IMREAD_ANYDEPTH),0.0,0.0,0.0,1e-8,1e-8,1e-8,0.0,0.0)
    #200 pixels
    #init map
    for i in range(200):
        mapper.map(batch_size=200,active_sampling=True)
    
    #camera pose
    last_pose=tracking_camera.params
    camera_vel=torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0]).detach().cuda().requires_grad_(True)
    last_kf=0
    mapping_thread=threading.Thread(target=mapping_thread,args=(mapper,))
    mapping_thread.start()
    for frame in range(1,frame_length):
        tracking_camera.params.data+=camera_vel
        tracking_camera.set_images(cv2.imread(rgb_files[frame],cv2.IMREAD_COLOR),cv2.imread(depth_files[frame],cv2.IMREAD_ANYDEPTH))
        pos=mappper.track(tracking_camera)
        camera_vel=0.2*camera_vel+0.8*(tracking_camera.params-last_pose)
        last_pose=tracking_camera.params
        if p < 0.65 and frame-last_kf>5:
            p=tracking_camera.params
            mapper.add_camera(rgb_files[frame],depth_filenames[frame],last_pose[3],last_pose[4],last_pose[5],last_pose[0],last_pose[1],last_pose[2],last_pose[6],last_pose[7])
            print("addded keyframe")
            print(last_pose)
            last_kf=frame
        mapper.render(tracking_camera,"view")
    mapping_thread.join()


main()