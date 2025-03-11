import numpy as np
import open3d as o3d

# -------- Step 1: Load the full point cloud --------
pc_np = np.load('/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/pointcloud/pointcloud_gq/point_cloud_gq.npy')
# If loaded as a structured array (with fields 'x', 'y', 'z'), convert it.
if pc_np.dtype.names is not None:
    pc_np = np.vstack([pc_np[name] for name in ('x', 'y', 'z')]).T

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc_np)

# -------- Step 2: Remove ground by thresholding in z --------
z_values = np.asarray(pcd.points)[:, 2]
ground_threshold = np.percentile(z_values, 10)
margin = 0.05  # 5 cm margin above ground
print(f"Ground threshold (10th percentile): {ground_threshold:.2f}")

non_ground_indices = np.where(z_values > (ground_threshold + margin))[0]
non_ground_points = np.asarray(pcd.points)[non_ground_indices]
non_ground_pcd = o3d.geometry.PointCloud()
non_ground_pcd.points = o3d.utility.Vector3dVector(non_ground_points)
print(f"Non-ground point cloud has {len(non_ground_points)} points.")

# -------- Step 3: Cluster the non-ground points using DBSCAN --------
# Adjusted parameters: increase eps and lower min_points.
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Info) as cm:
    labels = np.array(non_ground_pcd.cluster_dbscan(eps=0.3, min_points=5, print_progress=True))
max_label = labels.max()
print(f"Detected {max_label + 1} clusters (ignoring noise with label -1).")

# -------- Step 4: For each cluster, compute convex hull and fit it to new ground --------
convex_hull_meshes = []
for cluster_label in range(max_label + 1):
    indices = np.where(labels == cluster_label)[0]
    if len(indices) < 3:
        continue  # Skip clusters that are too small.
    cluster_pcd = non_ground_pcd.select_by_index(indices)
    
    try:
        hull, _ = cluster_pcd.compute_convex_hull()
    except Exception as e:
        print(f"Convex hull computation failed for cluster {cluster_label}: {e}")
        continue
    
    # Shift the convex hull so that its minimum z becomes 0.
    hull_points = np.asarray(hull.vertices)
    min_z = hull_points[:, 2].min()
    hull_points[:, 2] -= min_z
    hull.vertices = o3d.utility.Vector3dVector(hull_points)
    
    # Paint the hull with a random color.
    color = np.random.rand(3)
    hull.paint_uniform_color(color)
    convex_hull_meshes.append(hull)

# -------- Step 5: Visualize the convex hull objects in 3D using Open3D --------
if convex_hull_meshes:
    o3d.visualization.draw_geometries(convex_hull_meshes,
                                      window_name="Convex Hulls of Clusters Fitted to New Ground",
                                      width=800, height=600)
else:
    print("No convex hulls to display. Consider further adjusting clustering parameters.")
