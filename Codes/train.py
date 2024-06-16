import os
import torch
import logging
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch3d.loss import chamfer_distance
from model import seedformer_dim128
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import utils.utils as utils
from utils.utils import get_loss


# from dataset import CustomDataset  # Import your custom dataset class here
class RacingDataset(Dataset):
    def __init__(self, root_dir):  # 4731
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        self.filter_file_list = self.filter_list()
        # self.target_points = target_points

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
logging.basicConfig(filename='training_1305v002.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
epochs = 200
# Set random seed for reproducibility
torch.manual_seed(42)

# Initialize your model
model = seedformer_dim128(up_factors=[1, 2, 4,4])  # 512 1024 4096 9192

# Define device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
model.to(device)

# Define DataLoader
# Replace CustomDataset with your actual dataset class and provide necessary arguments
dataset = RacingDataset(root_dir="/dev/shm/IAC/SeedFormer_2602_npy/cropped/real")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8, collate_fn=utils.collate_fn)
# Define optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=3e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)
# scheduler = StepLR(optimizer, step_size=20, gamma=0.1)  # Reduce LR by a factor of 0.1 every 5 epochs

# Define paths for checkpoint
checkpoint_path = 'training2205v002_pad3000_knn_big.pth'

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
    logging.info(model)
# Training loop
train_losses = []
for epoch in range(start_epoch, epochs):
    running_loss = 0

   # total_cd_pc = 0
   # total_cd_p1 = 0
   # total_cd_p2 = 0
   # total_cd_p3 = 0
   # total_partial = 0
#
    # Wrap the DataLoader with tqdm to track progress
    with tqdm(enumerate(dataloader, 0), total=len(dataloader), desc=f'Epoch {epoch + 1}/{epochs}',
              unit='batch') as pbar:
        for i, data in pbar:
            inputs, labels = data
            inputs = inputs.to(device)  # Move data to GPU if available

            optimizer.zero_grad()

            # Apply the model to the entire batch
            outputs = model(inputs)

            for input_pc, output_pc in zip(inputs, outputs[-1]):
                loss, _ = chamfer_distance(input_pc.unsqueeze(0), output_pc.unsqueeze(0))
                losses.append(loss*1e3)
            losses = []

            loss = torch.mean(torch.stack(losses))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (i + 1))  # Update tqdm progress bar with the current loss

            # Compute losses
            #loss_total, losses, gts = get_loss(outputs, inputs, inputs, sqrt=False)
#
            ## Backpropagation and optimization
            #loss_total.backward()
            #optimizer.step()
#
            ## Log the loss for monitoring
            #running_loss += loss_total.item()
#
            #cd_pc_item = losses[0].item()
            #total_cd_pc += cd_pc_item
            #cd_p1_item = losses[1].item()
            #total_cd_p1 += cd_p1_item
            #cd_p2_item = losses[2].item()
            #total_cd_p2 += cd_p2_item
            #cd_p3_item = losses[3].item()
            #total_cd_p3 += cd_p3_item
            #partial_item = losses[4].item()
            #total_partial += partial_item

            # Compute average losses for the batch
           # avg_cdc = total_cd_pc / len(inputs)
           # avg_cd1 = total_cd_p1 / len(inputs)
           # avg_cd2 = total_cd_p2 / len(inputs)
           # avg_cd3 = total_cd_p3 / len(inputs)
           # avg_partial = total_partial / len(inputs)
#
            # Update running loss
            # print("len inputs" , len(inputs))
            #  print("loss ",loss_total/len(inputs))
            # Update progress bar with the current loss

        avg_loss = running_loss / len(dataloader)
        scheduler.step(avg_loss)  # Use the average loss for the scheduler

    # Compute average loss for the epoch
    avg_loss = running_loss / len(dataloader)
    train_losses.append(avg_loss)

    # Save checkpoint if current loss is the best seen so far
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)

    # Log the epoch loss
    last_lr = optimizer.param_groups[0]['lr']  # Get the last learning rate
  #  logging.info(   f'Epoch {epoch + 1} Loss: {loss_total, avg_cd1, avg_cd2, avg_cd3, avg_partial} LR: {last_lr}')  # Log it into the file
    logging.info(f'Epoch {epoch + 1} Loss: {avg_loss} LR: {last_lr}')
    train_losses.append(avg_loss)
logging.log(train_losses)
print('Finished Training')
