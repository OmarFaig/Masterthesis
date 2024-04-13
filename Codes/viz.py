import os
import numpy as np
import open3d as o3d

# Initialize frame index
frame_index = 0

# Path to folder containing .npy files for point clouds
pc_folder_path = '/home/omar/TUM/Data/SeedFormer_2602_npy/reconstructed_1004_100v001_pad10000/points'

# Path to folder containing bounding box text files
bbox_folder_path = '/home/omar/TUM/Data/SeedFormer_2602_npy/reconstructed_1004_100v001_pad10000/labels'

# List all .npy files in the folder
file_list = sorted([f for f in os.listdir(pc_folder_path) if f.endswith(('.npy','.pcd'))])

print(file_list)
# Create Open3D visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

vis.get_render_option().point_size = 3.0
vis.get_render_option().background_color = np.zeros(3)
#view_control = vis.get_view_control()

#view_control.set_up([0, 1, 0])
#ctr = vis.get_view_control()
#parameters = ctr.convert_to_pinhole_camera_parameters()
#parameters.extrinsic = np.array([[1, 0, 0, 0],
                          #      [1, 0, 0, 0],  # Look from the side, flipping y-axis
                          #      [1, 0, 0, 0],
                          #      [1, 0, 0, 0]])  # Translation remains unchanged
#ctr.convert_from_pinhole_camera_parameters(parameters)
# Lock the view up direction along the z-axis

# view_control.set_lookat(bbox.get_center())
#
#view_control.set_lookat(np.array([7.0, 8.0, 100.0]))
#view_control.scale(9)  # Adjust the zoom level as needed to zoom in closer
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


    bbox_file_path = os.path.join(bbox_folder, file_name.replace('.npy', '.txt').replace('.pcd', '.txt'))
    print(bbox_file_path)

    with open(bbox_file_path, 'r') as file:
        line = file.readline().strip()  # Read the single line from the file
        bbox_coordinates = line.split()  # Split the line into label and coordinates
        print(bbox_coordinates)
        # Check if the label is "Car", if yes, discard it

        bbox_coordinates = bbox_coordinates[:-1]  # Exclude the label

        bbox_coordinates = list(map(float, bbox_coordinates))  # Convert coordinates to float and then to NumPy array


    return point_cloud, bbox_coordinates


# Function to visualize point cloud and bounding box
def visualize(point_cloud, bbox_coordinates):
    # Create Open3D point cloud


    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(point_cloud)
   # view_control.set_zoom(125)

    # Create Open3D bounding box
   # if bbox_coordinates is not None:
    bbox = o3d.geometry.OrientedBoundingBox(center=bbox_coordinates[:3],
                                            R=np.eye(3),
                                            extent=bbox_coordinates[3:6])
    print(bbox.get_center())
    #view_control.set_lookat(bbox.get_center())
    bbox.color = [1, 0, 0]  # Set bbox color to red
   # else:

      #  bbox = None

    # Visualize point cloud and bounding box
    #o3d.visualization.draw_geometries([pcl, bbox])
    vis.clear_geometries()



    vis.add_geometry(pcl)
    vis.add_geometry(bbox)
    view_control = vis.get_view_control()

    view_control.set_front([0.1, -0.5, 0.6])
    view_control.set_lookat(bbox.get_center())
    #view_control.set_up([-0.36828927493940194, 0.49961995188329117, 0.78405542766104697])
    view_control.set_zoom(0.009)
def load_next_frame(vis):
    global frame_index
    print("Loading frame", frame_index)

    vis.clear_geometries()
    # Set the camera view to look at the center of the bounding box

    if frame_index < len(file_list):
        point_cloud, bbox_coordinates = load_data(pc_folder_path, bbox_folder_path, file_list[frame_index])
        if point_cloud is not None:
            visualize(point_cloud, bbox_coordinates)
        frame_index += 1
    #view_control.set_lookat(bbox_coordinates[:3])


def load_prev_frame(vis):
    global frame_index
    if frame_index >= 2:
        frame_index -= 2
        return load_next_frame(vis)
    else:
        frame_index = 0
        return load_next_frame(vis)

def close_vis(vis):
    vis.close()
    vis.destroy_window()



# Register key callbacks
key_to_callback = {}
key_to_callback[ord("N")] = load_next_frame
key_to_callback[ord("B")] = load_prev_frame
key_to_callback[ord("X")] = close_vis

load_next_frame(vis)

for key, val in key_to_callback.items():
    vis.register_key_callback(key, val)

vis.run()
