import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2  # For image dilation
import sensor_msgs_py.point_cloud2 as pc2

# --- Step 1: Load the point cloud from the .npy file ---
pc_np = np.load('point_cloud_gq.npy')

# Convert a structured array (if necessary) to a regular (N, 3) float array.
if pc_np.dtype.names is not None:
    # Assumes field names 'x', 'y', and 'z' exist.
    pc_np = np.vstack([pc_np[name] for name in ('x', 'y', 'z')]).T

# Create an Open3D PointCloud object.
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc_np)

# --- Step 2: Segment the ground plane using RANSAC ---
plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Detected ground plane: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# --- Step 3: Remove the ground points ---
# Invert the inlier selection to keep only non-ground points.
non_ground_cloud = pcd.select_by_index(inliers, invert=True)
print(f"Non-ground point cloud has {len(non_ground_cloud.points)} points.")

# --- Step 4: Flatten the remaining points onto the x-y plane ---
non_ground_np = np.asarray(non_ground_cloud.points)
occupancy_points = non_ground_np[:, :2]  # Only x and y

# --- Step 5: Create a 2D occupancy map ---
# Define grid parameters for the occupancy map.
resolution = 0.05  # grid cell size (e.g., 5 cm)
min_xy = occupancy_points.min(axis=0)
max_xy = occupancy_points.max(axis=0)
print(f"Occupancy map range: min {min_xy}, max {max_xy}")

# Create bins for the x and y axes.
x_bins = np.arange(min_xy[0], max_xy[0] + resolution, resolution)
y_bins = np.arange(min_xy[1], max_xy[1] + resolution, resolution)

# Create a 2D histogram; each cell counts the number of points falling into it.
H, xedges, yedges = np.histogram2d(occupancy_points[:, 0],
                                   occupancy_points[:, 1],
                                   bins=[x_bins, y_bins])

# Save the raw occupancy map.
np.save('occupancy_map_raw.npy', H)
print("Saved raw occupancy map to 'occupancy_map_raw_gq.npy'")

# --- Step 6: Apply dilation to emphasize walls ---
# Convert the occupancy map to a binary image: cells with > 0 points become 1.
binary_map = (H > 0).astype(np.uint8) * 255  # scale to 0-255 for OpenCV
# Define a dilation kernel (you can adjust the size to change the dilation effect).
kernel = np.ones((5, 5), np.uint8)
# Apply dilation.
dilated_map = cv2.dilate(binary_map, kernel, iterations=1)
# Save the dilated occupancy map.
np.save('occupancy_map_dilated.npy', dilated_map)
print("Saved dilated occupancy map to 'occupancy_map_dilated_gq.npy'")

# --- Step 7: Visualize the results using matplotlib ---
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Plot the raw occupancy map.
im0 = ax[0].imshow(binary_map.T, origin='lower', cmap='gray',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax[0].set_title("Raw Occupancy Map")
ax[0].set_xlabel("X (forward)")
ax[0].set_ylabel("Y (left)")
fig.colorbar(im0, ax=ax[0], label="Occupied")

# Plot the dilated occupancy map.
im1 = ax[1].imshow(dilated_map.T, origin='lower', cmap='gray',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax[1].set_title("Dilated Occupancy Map")
ax[1].set_xlabel("X (forward)")
ax[1].set_ylabel("Y (left)")
fig.colorbar(im1, ax=ax[1], label="Occupied")

plt.tight_layout()
plt.savefig('occupancy_maps.png', dpi=300)
plt.show()
