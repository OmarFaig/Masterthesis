import torch
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import open3d as o3d
import os
class RacingDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        self.filter_file_list = self.filter_list()

    def __len__(self):
        return len(self.filter_file_list)

    def __getitem__(self,index):
        pcd_path = os.path.join(self.root_dir,self.filter_file_list[index])
        pcd = o3d.io.read_point_cloud(pcd_path)

        points = torch.tensor(pcd.points, dtype=torch.float32)

        return points,pcd_path

    def filter_list(self):
        '''
        Filter the inputs so that only pcds with more than 50 points are included in the training
        :return:
        '''
        filtered_list=[]
        for filename in self.file_list:
            pcd = o3d.io.read_point_cloud(os.path.join(self.root_dir,filename))
            points = torch.tensor(pcd.points, dtype=torch.float32)
            if len(points)>=50:
                filtered_list.append(filename)
        return filtered_list
#dataset = RacingDataset(root_dir="/home/omar/TUM/Data/cropped/sim")
#dataloader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=2)

#for i in dataset:
#    #pcd = o3d.io.read_point_cloud(i)
#    print(i.shape)
#
#print(len(dataloader))
#