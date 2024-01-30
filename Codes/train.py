from model import seedformer_dim128
from utils.dataloader import RacingDataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# parser and cfg

dataset = RacingDataset(root_dir="/home/omar/TUM/Data/cropped/real")
dataloader = DataLoader(dataset,batch_size=2,shuffle=False,num_workers=2)
model = seedformer_dim128(up_factors=[1, 2, 2])
model = model.cuda()
num_points_pcd={}
#print(model)
for i ,x in dataset:
    print(i.shape[0]," - ", x)
    num_points_pcd[i.shape[0]]=x
    i = i.unsqueeze(0)
    i = i.cuda()
    y = model(i)

print(num_points_pcd)