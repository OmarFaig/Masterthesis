import open3d as o3d
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import os
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points, sample_farthest_points


def visualize(pcl, bbox_coordinates):
    '''
    Function for visualizing the point cloud and bounding box.

    :param pcl: The point cloud to visualize.
    :param bbox_coordinates: Coordinates of the bounding box.
    :return: Visualized point cloud.
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
    # opt.show_coordinate_frame = True
    # opt.background_color = np.asarray([0, 0, 0])
    view_control = vis.get_view_control()
    view_control.set_up([0, 1, 0])  # Lock the view up direction along the z-axis
    view_control.set_front([0.1, -0.5, 0.6])
    view_control.set_lookat(bbox.get_center())
    # view_control.set_up([-0.36828927493940194, 0.49961995188329117, 0.78405542766104697])
    view_control.set_zoom(0.009)
    bbox_line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
    # Set color and thickness for bounding box lines
    color = (1, 0, 0)  # Red color
    line_width = 10  # Thickness of lines

    # Assign colors to the LineSet
    bbox_line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(bbox_line_set.lines))])

    # Set line width for bounding box lines
    # for line in bbox_line_set.lines:
    #    line.points = line.points * line_width
    # o3d.visualization.draw_geometries([pcl, bbox_line_set])
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


def crop_bbox(pcd, bbox_coordinates, save_path):  # num_random_points):
    '''
    Crop the bounding box out of the point cloud and save it.

    :param pcd: The original point cloud.
    :param bbox_coordinates: Coordinates of the bounding box.
    :param save_path: Path to save the cropped point cloud.
    :return: None
    '''
    orientation_angle = float(bbox_coordinates[-1])
    h, w, l = bbox_coordinates[8:11]
    rotation_mat_z = np.array([
        [np.cos(orientation_angle), -np.sin(orientation_angle), 0.0],
        [np.sin(orientation_angle), np.cos(orientation_angle), 0.0],
        [0.0, 0.0, 1.0]
    ])
    center = bbox_coordinates[11:14]
    bbox = o3d.geometry.OrientedBoundingBox(center=center, R=rotation_mat_z, extent=[l, w, h])
    bbox_crop = pcd.crop(bbox)
    # print(save_path)
    o3d.io.write_point_cloud(save_path, bbox_crop)
    bbox_line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
    color = (1, 0, 0)

    colors = [color for _ in range(len(bbox_line_set.lines))]

    # Assign colors to the LineSet
    bbox_line_set.colors = o3d.utility.Vector3dVector(colors)


def crop_invert_stitch(original_pcd, car_reconstructed, bbox_coords):
    '''
    Crop the bounding box from the original point cloud, invert the selection, and stitch with the reconstructed car point cloud.

    :param original_pcd: The original point cloud.
    :param car_reconstructed: The reconstructed car point cloud.
    :param bbox_coords: Coordinates of the bounding box.
    :return: Stitched point cloud.
    '''
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
    # original_crop_invert = o3d.geometry.PointCloud.crop(original_pcd, bbox)
    inliers_indices = bbox.get_point_indices_within_bounding_box(original_pcd.points)

    # inliers_pcd = original_pcd.select_by_index(inliers_indices, invert=False)  # select inside points = cropped
    outliers_pcd = original_pcd.select_by_index(inliers_indices, invert=True)  # select outside points
    stitched_pcd = outliers_pcd + car_reconstructed

    return stitched_pcd


def correct_bbox_label(bbox_list):
    '''
    Adjust the given bounding box coordinates according to KITTI format.

    :param bbox_list: List of bounding box coordinates.
    :return: Corrected bounding box coordinates.
    '''
    name = bbox_list.strip().split()  # 'Car'

    del name[1:8]  # delete unnecessary zeros in between
    name.append(name.pop(0))  # append 'Car' to the back
    dx, dy, dz = name[0:3]
    temp = dx
    dx = dz
    dz = temp
    x, y, z = name[3:6]

    name[3:6] = dx, dy, dz  # replace dx and dz

    name[0:3] = x, y, z  # just move x,y,z to the front of the list

    return name


def collate_fn(batch):
    '''
    Custom collate function for DataLoader to pad each point cloud to a target number of points.

    :param batch: Batch of data containing point clouds and paths.
    :return: Padded point clouds and paths.
    '''
    # Define the target number of points for padding
    target_num_points = 5294

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


def chamfer(p1, p2):
    '''
    Compute the Chamfer distance between two point clouds.

    :param p1: First point cloud.
    :param p2: Second point cloud.
    :return: Chamfer distance.
    '''
    d1, _ = chamfer_distance(p1, p2)
    return torch.mean(d1)


def chamfer_sqrt(p1, p2):
    '''
    Compute the square root of the Chamfer distance between two point clouds.

    :param p1: First point cloud.
    :param p2: Second point cloud.
    :return: Square root of the Chamfer distance.
    '''
    d1, d2 = chamfer_distance(p1, p2)
    d1 = torch.clamp(d1, min=1e-9)
    # d2 = torch.clamp(d2, min=1e-9)
    # d1 = torch.mean(torch.sqrt(d1))
    # d2 = torch.mean(torch.sqrt(d2))
    return d1


def chamfer_single_side(pcd1, pcd2):
    '''
    Compute the single-sided Chamfer distance from pcd1 to pcd2.

    :param pcd1: First point cloud.
    :param pcd2: Second point cloud.
    :return: Single-sided Chamfer distance.
    '''
    d1, _ = chamfer_distance(pcd1, pcd2)
    return d1


def chamfer_single_side_sqrt(pcd1, pcd2):
    '''
    Compute the single-sided square root Chamfer distance from pcd1 to pcd2.

    :param pcd1: First point cloud.
    :param pcd2: Second point cloud.
    :return: Single-sided square root Chamfer distance.
    '''
    d1, _ = chamfer_distance(pcd1, pcd2)
    d1 = torch.clamp(d1, min=1e-9)
    # d1 = torch.mean(torch.sqrt(d1))
    return d1


def get_loss(pcds_pred, partial, gt, sqrt=True):
    """
    Compute the loss function for the given predicted point clouds, partial point cloud, and ground truth point cloud.

    :param pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...].
    :param partial: Partial point cloud.
    :param gt: Ground truth point cloud.
    :param sqrt: Whether to compute the square root of the Chamfer distance. Default is True.
    :return: Tuple containing the total loss, individual losses, and sampled ground truth point clouds.
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side
    else:
        CD = chamfer_sqrt
        PM = chamfer_single_side

    Pc, P1, P2, P3 = pcds_pred

    gt_2, _ = sample_farthest_points(gt, K=P2.size()[1])
    gt_1, _ = sample_farthest_points(gt_2, K=P1.size()[1])
    gt_c, _ = sample_farthest_points(gt_1, K=Pc.size()[1])

    cdc = CD(Pc, gt_c)
    cd1 = CD(P1, gt_1)
    cd2 = CD(P2, gt_2)
    cd3 = CD(P3, gt)

    partial_matching = PM(partial, P3)

    loss_all = (cdc + cd1 + cd2 + cd3 + partial_matching) * 1e-1
    losses = [cdc, cd1, cd2, cd3, partial_matching]
    return loss_all, losses, [gt_2, gt_1, gt_c]


def apply_and_save_res(dataset, dataloader, model, savedir):
    """
    Apply the model to the dataset, save the reconstructed point clouds, and visualize the differences.

    :param dataset: The dataset containing the point clouds.
    :param dataloader: The DataLoader for iterating over the dataset.
    :param model: The model used for reconstructing the point clouds.
    :param savedir: The directory to save the reconstructed point clouds.
    :return: None
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    for dataset, dataloader in [(dataset, dataloader)]:
        for i, data in enumerate(tqdm(dataloader, unit='point cloud')):
            inputs, paths = data
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(inputs)

            outputs_cpu = outputs[-1].cpu().numpy()

            filename = os.path.basename(paths[0])
            output_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(outputs_cpu[0]))
            f_name = os.path.join(savedir, filename)
            o3d.io.write_point_cloud(f_name, output_pcd)
