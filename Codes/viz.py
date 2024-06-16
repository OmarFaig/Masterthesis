import os
import numpy as np
import open3d as o3d

# Initialize frame index
frame_index = 0

# Path to folder containing .npy files for point clouds
pc_folder_path = '/home/omar/TUM/Data/SeedFormer_2602_npy/reconstructed_2205v002_pad3000_knn_big/points'

# Path to folder containing bounding box text files
bbox_folder_path = '/home/omar/TUM/Data/SeedFormer_2602_npy/reconstructed_1105v004/labels'

# List all .npy and .pcd files in the folder
file_list = sorted([f for f in os.listdir(pc_folder_path) if f.endswith(('.npy', '.pcd'))])

print(file_list)

# Create Open3D visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

# Set rendering options
vis.get_render_option().point_size = 3.0
vis.get_render_option().background_color = np.zeros(3)

# Function to load .npy point cloud and bounding box coordinates
def load_data(pc_folder, bbox_folder, file_name):
    if file_name.endswith('.npy'):
        # Load point cloud from .npy file
        point_cloud = np.load(os.path.join(pc_folder, file_name))
    elif file_name.endswith('.pcd'):
        # Load point cloud from .pcd file using Open3D
        pcd_path = os.path.join(pc_folder, file_name)
        print(pcd_path)
        pcd = o3d.io.read_point_cloud(pcd_path)
        point_cloud = np.asarray(pcd.points)

    # Load bounding box coordinates from corresponding text file
    bbox_file_path = os.path.join(bbox_folder, file_name.replace('.npy', '.txt').replace('.pcd', '.txt'))
    print(bbox_file_path)

    with open(bbox_file_path, 'r') as file:
        line = file.readline().strip()
        bbox_coordinates = line.split()
        bbox_coordinates = bbox_coordinates[:-1]  # Exclude the label
        bbox_coordinates = list(map(float, bbox_coordinates))  # Convert coordinates to float

    return point_cloud, bbox_coordinates

# Function to visualize point cloud and bounding box
def visualize(point_cloud, bbox_coordinates):
    # Create Open3D point cloud
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(point_cloud)

    # Create Open3D bounding box
    bbox = None
    if bbox_coordinates is not None:
        bbox = o3d.geometry.OrientedBoundingBox(center=bbox_coordinates[:3],
                                                R=np.eye(3),
                                                extent=bbox_coordinates[3:6])
        bbox.color = [1, 0, 0]  # Set bbox color to red

    # Add geometries to the visualizer
    vis.clear_geometries()
    vis.add_geometry(pcl)
    if bbox:
        vis.add_geometry(bbox)

    # Get the view control and set view parameters
    view_control = vis.get_view_control()
    view_control.set_front([0.1, -0.5, 0.6])
    view_control.set_lookat(bbox.get_center())
    view_control.set_zoom(0.009)

# Function to load the next frame
def load_next_frame(vis):
    global frame_index
    print("Loading frame", frame_index)

    vis.clear_geometries()

    if frame_index < len(file_list):
        point_cloud, bbox_coordinates = load_data(pc_folder_path, bbox_folder_path, file_list[frame_index])
        if point_cloud is not None:
            visualize(point_cloud, bbox_coordinates)
        frame_index += 1

# Function to load the previous frame
def load_prev_frame(vis):
    global frame_index
    if frame_index >= 2:
        frame_index -= 2
        return load_next_frame(vis)
    else:
        frame_index = 0
        return load_next_frame(vis)

# Function to close the visualizer
def close_vis(vis):
    vis.close()
    vis.destroy_window()

# Register key callbacks
key_to_callback = {
    ord("N"): load_next_frame,
    ord("B"): load_prev_frame,
    ord("X"): close_vis
}

# Load the first frame
load_next_frame(vis)

# Register callbacks for each key
for key, val in key_to_callback.items():
    vis.register_key_callback(key, val)

vis.run()
