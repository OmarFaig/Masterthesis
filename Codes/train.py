from model import seedformer_dim128
from utils.dataloader import RacingDataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataset = RacingDataset(root_dir="/home/omar/TUM/Data/cropped/sim")
dataloader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=2)
#model = seedformer_dim128(up_factors=[1, 2, 2])
#model = model.cuda()
num_points_pcd={}
#print(model)
for i ,x in dataset:
    print(i.shape[0]," - ", x)
    #print(i.shape[0])
    num_points_pcd[i.shape[0]]=x
    #i = i.unsqueeze(0)
    ## x = torch.rand(8, 2048, 3)
    #i = i.cuda()
#
    #y = model(i)
    #print([pc.size() for pc in y])

print(num_points_pcd)
#plt.plot(num_points_pcd)
plt.plot(*zip(*sorted(num_points_pcd.items())))
plt.show()
#plt.scatter(num_points_pcd.keys(),num_points_pcd.values())
#plt.show()
# Replace this with your actual dictionary

# 1. Count files with less than 50 points
less_than_50_points = sum(1 for key in num_points_pcd if key < 50)
more_than_100_points = sum(1 for key in num_points_pcd if key > 100)
more_than_1000_points = sum(1 for key in num_points_pcd if key > 1000)
more_than_3000_points = sum(1 for key in num_points_pcd if key > 3000)

print(f"Number of files with less than 50 points: {less_than_50_points}")
print(f"Number of files with more than 100 points: {more_than_100_points}")
print(f"Number of files with more than 1000 points: {more_than_1000_points}")
print(f"Number of files with more than 3000 points: {more_than_3000_points}")

# 2. Visualize keys and values
keys = list(num_points_pcd.keys())
values = list(num_points_pcd.values())

plt.figure(figsize=(10, 6))
plt.bar(keys, keys, color='blue', alpha=0.7, label='Number of points')
plt.scatter(keys, [0] * len(keys), color='red', marker='o', label='File paths')

for key, value in zip(keys, values):
    plt.text(key, 0, f"\n{value}", rotation=45, ha='right', va='bottom')

plt.xlabel('Number of Points')
plt.title('Visualization of Number of Points in PCD Files')
plt.legend()
plt.show()
