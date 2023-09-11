# Dataloader for the DTU dataset in Yaoyao's format.
# by: Jiayu Yang
# date: 2020-01-28

# Note: This file use part of the code from the following projects.
#       Thanks for the authors for the great code.
#       MVSNet: https://github.com/YoYo000/MVSNet
#       MVSNet_pytorch: https://github.com/xy-guo/MVSNet_pytorch

from dataset.utils import *
from dataset.dataPaths import *
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image



# For debug:
# import matplotlib.pyplot as plt
# import pdb

class MVSDatasetColmap(Dataset):
    def __init__(self, args, logger=None):
        # Initializing the dataloader
        super(MVSDatasetColmap, self).__init__()
        
        # Parse input
        self.args = args
        self.data_root = self.args.dataset_root
        self.logger = logger
        if logger==None:
            import logger
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
            consoleHandler = logging.StreamHandler(sys.stdout)
            consoleHandler.setFormatter(formatter)
            self.logger.addHandler(consoleHandler)
            self.logger.info("File logger not configured, only writing logs to stdout.")
        self.logger.info("Initiating dataloader for our pre-processed DTU dataset.")
        self.logger.info("Using dataset:"+self.data_root+self.args.mode+"/")

        self.metas = self.build_list(self.args.mode)
        self.logger.info("Dataloader initialized.")

    def build_list(self,mode):
        # Build the item meta list
        metas = []
        image_dir=os.path.join(self.data_root,"images")
        names = os.listdir(image_dir)
        names.sort()

        kWinSize = 10 #滑动窗口的大小

        for i,name in enumerate(names):
            start_idx = i - int(kWinSize/2)
            end_idx = i + int(kWinSize/2)
            start_idx = max(0,start_idx)
            end_idx = min(len(names)-1,end_idx)

            ref_view = name
            src_views = [names[j] for j in range(start_idx,end_idx)]

            metas.append((ref_view, src_views))
        self.logger.info("Done. metas:"+str(len(metas)))
        return metas

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        ref_view, src_views = meta

        assert self.args.nsrc <= len(src_views)

        self.logger.debug("Getting Item:"+"\nref_view:"+str(ref_view)+"\nsrc_view:"+str(src_views))

        ref_img = [] 
        src_imgs = [] 
        ref_depths = [] 
        ref_depth_mask = [] 
        ref_intrinsics = [] 
        src_intrinsics = [] 
        ref_extrinsics = [] 
        src_extrinsics = [] 
        depth_min = [] 
        depth_max = [] 

        ## 1. Read images
        # ref image
        image_dir=os.path.join(self.data_root,"images")
        ref_img_file = os.path.join(image_dir,ref_view)
        ref_img = read_img(ref_img_file)

        # src image(s)
        for i in range(self.args.nsrc):
            src_img_file = os.path.join(image_dir,src_views[i])
            src_img = read_img(src_img_file)
            src_imgs.append(src_img)

        ## 2. Read camera parameters
        cam_dir=os.path.join(self.data_root,"cam")

        cam_file = os.path.join(cam_dir,ref_view+"_cam.txt")
        ref_intrinsics, ref_extrinsics, depth_min, depth_max = read_cam_file(cam_file)
        depth_min = 2
        depth_max = 50
        for i in range(self.args.nsrc):
            cam_file = os.path.join(cam_dir, src_views[i] + "_cam.txt")
            intrinsics, extrinsics, depth_min_tmp, depth_max_tmp = read_cam_file(cam_file)
            src_intrinsics.append(intrinsics)
            src_extrinsics.append(extrinsics)

        # Orgnize output and return
        sample = {}
        sample["ref_img"] = np.moveaxis(np.array(ref_img),2,0) # ndarray:{3,1184,1

        # }
        sample["src_imgs"] = np.moveaxis(np.array(src_imgs),3,1) # ndarray:{4,3,1184,1600}
        sample["ref_intrinsics"] = np.array(ref_intrinsics) # ndarray:{3,3}
        sample["src_intrinsics"] = np.array(src_intrinsics) # ndarray:{4,3,3}
        sample["ref_extrinsics"] = np.array(ref_extrinsics) # ndarray:{4,4}
        sample["src_extrinsics"] = np.array(src_extrinsics) # ndarray:{4,4,4}
        sample["depth_min"] = depth_min
        sample["depth_max"] = depth_max

        # print(sample)

        if self.args.mode == "train":
            sample["ref_depths"] = np.array(ref_depths,dtype=float)
            sample["ref_depth_mask"] = np.array(ref_depth_mask)
        elif self.args.mode == "test":
            sample["filename"] = '{}/' + '{}'.format(ref_view) + "{}"

        return sample

            

