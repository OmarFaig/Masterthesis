import torch
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import open3d as o3d
import os
class RacingDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,index):
        pcd_path = os.path.join(self.root_dir,self.file_list[index])
        pcd = o3d.io.read_point_cloud(pcd_path)

        points = torch.tensor(pcd.points, dtype=torch.float32)
        return points


dataset = RacingDataset(root_dir="/home/omar/TUM/Data/cropped/sim")
dataloader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=2)

for i in dataset:
    print(i.shape)

print(len(dataloader))
