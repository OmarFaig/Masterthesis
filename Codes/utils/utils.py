import open3d as o3d
import numpy as np

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
    opt.show_coordinate_frame = True
  #  opt.background_color = np.asarray([0, 0, 0])
    view_control = vis.get_view_control()
    view_control.set_up([0, 0, 1])  # Lock the view up direction along the z-axis
    #opt.show_grid=True
    bbox_line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
    color = (1, 0, 0)

    colors = [color for _ in range(len(bbox_line_set.lines))]

    # Assign colors to the LineSet
    bbox_line_set.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcl,bbox_line_set])
    vis.add_geometry(pcl)
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


def crop_invert_stitch(original_pcd, bbox_coords):
    '''
    Crop the bbox from  original pcd and return rest(not bbox)
    :param original_pcd:
    :param bbox_coords:
    :return:
    '''
    orientation_angle = float(bbox_coords[-1])
    h, w, l = bbox_coords[8:11]
    rotation_mat_z = np.array([
        [np.cos(orientation_angle), -np.sin(orientation_angle), 0.0],
        [np.sin(orientation_angle), np.cos(orientation_angle), 0.0],
        [0.0, 0.0, 1.0]
    ])
    center = bbox_coords[11:14]
    bbox = o3d.geometry.OrientedBoundingBox(center=center, R=rotation_mat_z, extent=[l, w, h])
    inliers_indices = bbox.get_point_indices_within_bounding_box(original_pcd.points)

    #inliers_pcd = original_pcd.select_by_index(inliers_indices, invert=False)  # select inside points = cropped
    outliers_pcd = original_pcd.select_by_index(inliers_indices, invert=True)  # select outside points

    return outliers_pcd

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