from model import seedformer_dim128
from utils.dataloader import RacingDataset
import torch
from torch.utils.data import DataLoader


dataset = RacingDataset(root_dir="/home/omar/TUM/Data/cropped/sim")
dataloader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=2)
model = seedformer_dim128(up_factors=[1, 2, 2])
model = model.cuda()
#print(model)
for i in dataset:
    print(i.shape)
    i = i.unsqueeze(0)
    # x = torch.rand(8, 2048, 3)
    i = i.cuda()

    y = model(i)
    print([pc.size() for pc in y])
