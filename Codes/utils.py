import open3d as o3d
import numpy as np

def visualize(pcl):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    # Call only after creating visualizer window.
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
  #  opt.background_color = np.asarray([0, 0, 0])
    view_control = vis.get_view_control()
    view_control.set_up([0, 0, 1])  # Lock the view up direction along the z-axis
    #opt.show_grid=True

    vis.add_geometry(pcl)
    vis.run()
    vis.destroy_window()

def crop_bbox(pcd,bbox_coordinates,save_path,num_random_points):
    orientation_angle=float(bbox_coordinates[-1])
    #print(orientation_angle)
    h,w,l=bbox_coordinates[8:11]
    rotation_mat_z=np.array([
        [np.cos(orientation_angle),-np.sin(orientation_angle),0.0],
        [np.sin(orientation_angle),np.cos(orientation_angle),0.0],
        [0.0,0.0,1.0]
    ])
    center=bbox_coordinates[11:14]
    bbox=o3d.geometry.OrientedBoundingBox(center=center,R=rotation_mat_z,extent=[ l,w,h])
    bbox_crop=pcd.crop(bbox)
    # generate random points uniformly
    random_points=np.random.uniform(-0.5,0.5,size=(num_random_points,3))
    random_points= np.dot(random_points,bbox.R.T)+bbox.center
    combined_points=np.vstack((np.asarray(bbox_crop.points),random_points))
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
 #  o3d.io.write_point_cloud(save_path,combined_pcd)

    o3d.io.write_point_cloud(save_path,bbox_crop)
    bbox_line_set=o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
    color=(1, 0, 0)

    colors = [color for _ in range(len(bbox_line_set.lines))]

    # Assign colors to the LineSet
    bbox_line_set.colors = o3d.utility.Vector3dVector(colors)


   # o3d.visualization.draw_geometries([pcd,bbox_line_set])
