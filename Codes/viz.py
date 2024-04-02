import os
import numpy as np
import open3d as o3d

# Initialize frame index
frame_index = 0

# Path to folder containing .npy files for point clouds
pc_folder_path = '/home/omar/TUM/Data/SeedFormer_2602_npy/reconstructed_0104/points'

# Path to folder containing bounding box text files
bbox_folder_path = '/home/omar/TUM/Data/SeedFormer_2602_npy/reconstructed_0104/labels'

# List all .npy files in the folder
file_list = sorted([f for f in os.listdir(pc_folder_path) if f.endswith('.npy')])


# Function to load .npy point cloud and bounding box coordinates
def load_data(pc_folder, bbox_folder, file_name):
    point_cloud = np.load(os.path.join(pc_folder, file_name))
    bbox_file_path = os.path.join(bbox_folder, file_name.replace('.npy', '.txt'))

    # Check if bbox file exists
    if os.path.exists(bbox_file_path):
        with open(bbox_file_path, 'r') as file:
            line = file.readline().strip()  # Read the single line from the file
            bbox_coordinates = line.split()  # Split the line into label and coordinates
            print(bbox_coordinates)
            # Check if the label is "Car", if yes, discard it

            bbox_coordinates = bbox_coordinates[:-1]  # Exclude the label
            print(bbox_coordinates)

            bbox_coordinates = np.array(
                list(map(float, bbox_coordinates)))  # Convert coordinates to float and then to NumPy array
    else:
        bbox_coordinates = None

    return point_cloud, bbox_coordinates


# Function to visualize point cloud and bounding box
def visualize(point_cloud, bbox_coordinates):
    # Create Open3D point cloud
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(point_cloud)

    # Create Open3D bounding box
    if bbox_coordinates is not None:
        bbox = o3d.geometry.OrientedBoundingBox(center=bbox_coordinates[:3],
                                                R=np.eye(3),
                                                extent=bbox_coordinates[3:6])
        bbox.color = [1, 0, 0]  # Set bbox color to red
    else:
        bbox = None

    # Visualize point cloud and bounding box
    o3d.visualization.draw_geometries([pcl, bbox])


def load_next_frame(vis):
    global frame_index
    if frame_index < len(file_list):
        point_cloud, bbox_coordinates = load_data(pc_folder_path, bbox_folder_path, file_list[frame_index])
        if point_cloud is not None:
            visualize(point_cloud, bbox_coordinates)
        frame_index += 1


def load_prev_frame(vis):
    global frame_index
    if frame_index > 0:
        frame_index -= 1
        point_cloud, bbox_coordinates = load_data(pc_folder_path, bbox_folder_path, file_list[frame_index])
        if point_cloud is not None:
            visualize(point_cloud, bbox_coordinates)


def close_vis(vis):
    vis.close()
    vis.destroy_window()


# Create Open3D visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

# Register key callbacks
key_to_callback = {ord('N'): load_next_frame, ord('B'): load_prev_frame, ord('X'): close_vis}
for key, val in key_to_callback.items():
    vis.register_key_callback(key, val)

# Load and visualize the initial frame
load_next_frame(vis)

# Run Open3D visualizer
vis.run()
