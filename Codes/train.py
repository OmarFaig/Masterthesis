import os
import torch
import logging
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch3d.loss import chamfer_distance
from model import seedformer_dim128
from torch.utils.data import Dataset,DataLoader
import open3d as o3d
import utils.utils as utils

#from dataset import CustomDataset  # Import your custom dataset class here
class RacingDataset(Dataset):
    def __init__(self, root_dir, target_points=2990):  # 4731
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        self.filter_file_list = self.filter_list()
        self.target_points = target_points

    def __len__(self):
        return len(self.filter_file_list)

    def __getitem__(self, index):
        pcd_path = os.path.join(self.root_dir, self.filter_file_list[index])
        pcd = o3d.io.read_point_cloud(pcd_path)

        points = torch.tensor(pcd.points, dtype=torch.float32)

        return points, pcd_path

    def filter_list(self):
        '''
        Filter the inputs so that only pcds with more than 50 points are included in the training
        :return:
        '''
        filtered_list = []
        for filename in self.file_list:
            pcd = o3d.io.read_point_cloud(os.path.join(self.root_dir, filename))
            points = torch.tensor(pcd.points, dtype=torch.float32)
            if len(points) >= 0:
                filtered_list.append(filename)
        return filtered_list


# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
epochs=1
# Set random seed for reproducibility
torch.manual_seed(42)

# Initialize your model
model = seedformer_dim128(up_factors=[1, 2, 2])

# Define device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
model.to(device)

# Define DataLoader
# Replace CustomDataset with your actual dataset class and provide necessary arguments
dataset = RacingDataset(root_dir="/home/omar/TUM/Data/cropped/real", target_points=4000)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8,collate_fn=utils.collate_fn)
# Define optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce LR by a factor of 0.1 every 5 epochs

# Define paths for checkpoint
checkpoint_path = 'checkpoint_train.pth'

# Check if a checkpoint exists
if os.path.exists(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    logging.info(f"Checkpoint loaded. Resuming training from epoch {start_epoch}. Best loss so far: {best_loss}")
else:
    start_epoch = 0
    best_loss = float('inf')
    logging.info("No checkpoint found. Starting training from scratch.")

# Training loop
for epoch in range(start_epoch, epochs):
    running_loss = 0

    # Wrap the DataLoader with tqdm to track progress
    with tqdm(enumerate(dataloader, 0), total=len(dataloader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
        for i, data in pbar:
            inputs, labels = data
            inputs = inputs.to(device)  # Move data to GPU if available

            optimizer.zero_grad()
            outputs = model(inputs)
            losses = []

            for input_pc, output_pc in zip(inputs, outputs[-1]):
                loss, _ = chamfer_distance(input_pc.unsqueeze(0), output_pc.unsqueeze(0))
                losses.append(loss)

            loss = torch.mean(torch.stack(losses))
            loss.backward()
            optimizer.step()

            running_loss += loss
            pbar.set_postfix(loss=running_loss / (i + 1))  # Update tqdm progress bar with the current loss

        scheduler.step()  # Step the learning rate scheduler

    # Save checkpoint if current loss is the best seen so far
    if running_loss < best_loss:
        best_loss = running_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': running_loss
        }, checkpoint_path)

    # Log the epoch loss
    avg_loss = running_loss / len(dataloader)
    logging.info(f'Epoch {epoch + 1} Loss: {avg_loss}')

print('Finished Training')
