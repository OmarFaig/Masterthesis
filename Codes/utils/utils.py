import open3d as o3d
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import os
def visualize(pcl,bbox_coordinates):
    '''
    Function for visualizing the point cloud and bbox

    :param pcl:
    :param bbox_coordinates:
    :return: visualized point cloud
    '''
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    h, w, l = bbox_coordinates[8:11]
    orientation_angle = float(bbox_coordinates[-1])

    rotation_mat_z = np.array([
        [np.cos(orientation_angle), -np.sin(orientation_angle), 0.0],
        [np.sin(orientation_angle), np.cos(orientation_angle), 0.0],
        [0.0, 0.0, 1.0]
    ])
    center = bbox_coordinates[11:14]
    bbox = o3d.geometry.OrientedBoundingBox(center=center, R=rotation_mat_z, extent=[l, w, h])
    opt = vis.get_render_option()
    #opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0, 0, 0])
    view_control = vis.get_view_control()
    view_control.set_up([0, 1, 0])  # Lock the view up direction along the z-axis

    bbox_line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
    color = (0, 0, 0)

    colors = [color for _ in range(len(bbox_line_set.lines))]

    # Assign colors to the LineSet
    bbox_line_set.colors = o3d.utility.Vector3dVector(colors)

   # o3d.visualization.draw_geometries([pcl,bbox_line_set])
    vis.add_geometry(pcl)
    vis.add_geometry(bbox_line_set)
    ctr = vis.get_view_control()
    parameters = ctr.convert_to_pinhole_camera_parameters()
    parameters.extrinsic = np.array([[1, 0, 0, 0],
                                     [0, 1, -1, 0],  # Look from the side, flipping y-axis
                                     [0, 1, 0, 0],
                                     [0, 0, 0, 1]])  # Translation remains unchanged
    ctr.convert_from_pinhole_camera_parameters(parameters)
    vis.run()
    vis.destroy_window()

def crop_bbox(pcd,bbox_coordinates,save_path):#num_random_points):
    '''
    Crop the bbox out of pcd and save it
    :param pcd:
    :param bbox_coordinates:
    :param save_path:
    :return:
    '''
    orientation_angle=float(bbox_coordinates[-1])
    h,w,l=bbox_coordinates[8:11]
    rotation_mat_z=np.array([
        [np.cos(orientation_angle),-np.sin(orientation_angle),0.0],
        [np.sin(orientation_angle),np.cos(orientation_angle),0.0],
        [0.0,0.0,1.0]
    ])
    center = bbox_coordinates[11:14]
    bbox = o3d.geometry.OrientedBoundingBox(center=center,R=rotation_mat_z,extent=[ l,w,h])
    bbox_crop = pcd.crop(bbox)


    o3d.io.write_point_cloud(save_path,bbox_crop)
    bbox_line_set=o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
    color=(1, 0, 0)

    colors = [color for _ in range(len(bbox_line_set.lines))]

    # Assign colors to the LineSet
    bbox_line_set.colors = o3d.utility.Vector3dVector(colors)


def crop_invert_stitch(original_pcd, car_reconstructed, bbox_coords):
    orientation_angle = float(bbox_coords[-1])
    # print(orientation_angle)
    h, w, l = bbox_coords[8:11]
    rotation_mat_z = np.array([
        [np.cos(orientation_angle), -np.sin(orientation_angle), 0.0],
        [np.sin(orientation_angle), np.cos(orientation_angle), 0.0],
        [0.0, 0.0, 1.0]
    ])
    center = bbox_coords[11:14]
    bbox = o3d.geometry.OrientedBoundingBox(center=center, R=rotation_mat_z, extent=[l, w, h])
    # orignal_crop_invert =o3d.geometry.PointCloud.crop(original_pcd,bbox)
    inliers_indices = bbox.get_point_indices_within_bounding_box(original_pcd.points)

    inliers_pcd = original_pcd.select_by_index(inliers_indices, invert=False)  # select inside points = cropped
    outliers_pcd = original_pcd.select_by_index(inliers_indices, invert=True)  # select outside points
    stitched_pcd = outliers_pcd + car_reconstructed

    return stitched_pcd

def correct_bbox_label(bbox_list):
    '''
    adjust the given bbox coordinates acccording to KITTi format
    :param bbox_list:
    :return:
    '''
    name = bbox_list.strip().split()     #'Car'

    del name[1:8]    # delete unnecessary zeros inbetween
    name.append(name.pop(0))   # append 'Car' to the back
    dx, dy, dz = name[0:3]
    temp = dx
    dx = dz
    dz = temp
    x, y, z = name[3:6]

    name[3:6] = dx, dy, dz #  replce dx and dz

    name[0:3] = x, y, z  #  just move x,y,z to the front of the list

    return name

def collate_fn(batch):
    # Define the target number of points for padding
    target_num_points = 4731

    # Pad each point cloud to have target_num_points points
    padded_points_batch = []
    for points, _ in batch:
        num_points_to_pad = target_num_points - points.size(0)
        if num_points_to_pad > 0:
            pad = torch.zeros(num_points_to_pad, 3, dtype=torch.float32)
            padded_points = torch.cat((points, pad), dim=0)
        else:
            padded_points = points[:target_num_points]  # Trim to target_num_points if exceeds
        padded_points_batch.append(padded_points)

    padded_points = pad_sequence(padded_points_batch, batch_first=True, padding_value=0)

    paths = [item[1] for item in batch]

    return padded_points, paths


def apply_and_save_res(dataset, dataloader, model, savedir):
    model.eval()
    # Apply the model and visualize differences
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    for dataset, dataloader, in [(dataset, dataloader)]:
        for i, data in enumerate(tqdm(dataloader, unit='point cloud')):
            # if i >= num_samples:
            #    break

            inputs, paths = data
            # print(paths)
            inputs = inputs.to(device)

            with torch.no_grad():
                outputs = model(inputs)

            outputs_cpu = outputs.cpu().numpy()

            # for path, output in zip(paths, outputs_cpu):
            filename = os.path.basename(paths[0])
            output_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(outputs_cpu[0]))
            f_name = os.path.join(savedir, filename)
            o3d.io.write_point_cloud(f_name, output_pcd)
