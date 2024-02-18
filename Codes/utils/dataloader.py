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

    def get_largest_point_cloud(self):
        largest_pcd_path = None
        largest_num_points = 0
        for filename in self.filter_file_list:
            pcd = o3d.io.read_point_cloud(os.path.join(self.root_dir, filename))
            num_points = len(pcd.points)
            if num_points > largest_num_points:
                largest_num_points = num_points
                largest_pcd_path = os.path.join(self.root_dir, filename)
        print("Largest point cloudl", largest_num_points)
        print("Largest point cloudl", largest_pcd_path)

        return largest_pcd_path, largest_num_points
dataset = RacingDataset(root_dir="/home/omar/TUM/Data/cropped/sim")
dataset.get_largest_point_cloud()
dataloader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=2)

#for i in dataset:
#    #pcd = o3d.io.read_point_cloud(i)
#    print(i.shape)

#print(len(dataloader))

#