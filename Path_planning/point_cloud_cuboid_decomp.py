#!/usr/bin/env python3
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import random

# Additional imports for graph + plotting
import networkx as nx
import matplotlib.pyplot as plt

#############################################
# 1. Obstacle Expansion (3D Safety Margin)
#############################################
def expand_obstacles_3d(occupancy, safety_voxels=2):
    from scipy.ndimage import generate_binary_structure, binary_dilation
    structure = generate_binary_structure(rank=3, connectivity=1)  # 6-connected 3D structure
    expanded = binary_dilation(occupancy.astype(bool), structure=structure, iterations=safety_voxels)
    return expanded.astype(np.uint8)

#############################################
# 2. Build 3D Occupancy Grid from Point Cloud
#############################################
def build_occupancy_grid(points, global_min, resolution):
    extent = np.max(points, axis=0) - global_min
    nx = int(np.ceil(extent[0] / resolution))
    ny = int(np.ceil(extent[1] / resolution))
    nz = int(np.ceil(extent[2] / resolution))
    occupancy = np.zeros((nx, ny, nz), dtype=np.uint8)
    
    idxs = np.floor((points - global_min) / resolution).astype(int)
    idxs = np.clip(idxs, 0, [nx-1, ny-1, nz-1])
    for (ix, iy, iz) in idxs:
        occupancy[ix, iy, iz] = 1
    return occupancy

#############################################
# 3. Greedy 3D Region Growing for Free Cuboids
#    with a fixed max_z_thickness
#############################################
def region_growing_3d(occupancy, max_z_thickness=5):
    """
    Greedily grows free cuboids in X/Y, but caps the thickness in Z to 'max_z_thickness'.
    occupancy: 3D numpy array (1=obstacle, 0=free).
    max_z_thickness: the maximum allowed size in the z dimension for each cuboid.
    Returns a list of cuboids, each as a dictionary:
      {
         'min_idx': np.array([ix, iy, iz]),
         'dimensions': np.array([dx, dy, dz])
      }
    """
    nx, ny, nz = occupancy.shape
    visited = np.zeros_like(occupancy, dtype=bool)
    cuboids = []
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if occupancy[i, j, k] == 0 and not visited[i, j, k]:
                    # Initialize boundaries for region growing.
                    i_min, i_max = i, i
                    j_min, j_max = j, j
                    k_min, k_max = k, k
                    changed = True
                    while changed:
                        changed = False
                        # Expand in +x
                        if i_max < nx - 1:
                            candidate = occupancy[i_max+1, j_min:j_max+1, k_min:k_max+1]
                            if np.all(candidate == 0):
                                i_max += 1
                                changed = True
                        # Expand in +y
                        if j_max < ny - 1:
                            candidate = occupancy[i_min:i_max+1, j_max+1, k_min:k_max+1]
                            if np.all(candidate == 0):
                                j_max += 1
                                changed = True
                        # Expand in +z, but do not exceed max_z_thickness
                        current_z_thickness = (k_max - k_min + 1)
                        if current_z_thickness < max_z_thickness and k_max < nz - 1:
                            candidate = occupancy[i_min:i_max+1, j_min:j_max+1, k_max+1]
                            if np.all(candidate == 0):
                                k_max += 1
                                changed = True
                    visited[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1] = True
                    cuboid = {
                        'min_idx': np.array([i_min, j_min, k_min]),
                        'dimensions': np.array([i_max - i_min + 1, j_max - j_min + 1, k_max - k_min + 1])
                    }
                    cuboids.append(cuboid)
    return cuboids

#############################################
# 4. Convert Grid Cuboids to World Coordinates
#############################################
def block_to_world_cuboid(block, global_min, resolution):
    min_idx = block['min_idx']
    dims = block['dimensions']
    lower = global_min + (min_idx * resolution)
    upper = global_min + ((min_idx + dims) * resolution)
    return {
        'lower': lower,
        'upper': upper,
        'dimensions_world': upper - lower
    }

#############################################
# 5. ROS2 Publisher Node for RViz2 Visualization
#############################################
class RVizPublisher(Node):
    def __init__(self, points, cuboids, frame_id="map"):
        super().__init__('rviz_region_growing_publisher')
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pc_pub = self.create_publisher(PointCloud2, '/point_cloud', qos)
        self.marker_pub = self.create_publisher(MarkerArray, '/free_cuboids', qos)
        self.timer = self.create_timer(1.0, self.publish_all)
        
        self.points = points
        self.cuboids = cuboids
        self.frame_id = frame_id
        
        self.get_logger().info("3D Region Growing Publisher initialized.")
    
    def publish_all(self):
        now = self.get_clock().now().to_msg()
        self.publish_point_cloud(now)
        self.publish_cuboid_markers(now)
        self.get_logger().info("Published point cloud and cuboid markers.")
    
    def publish_point_cloud(self, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = self.frame_id
        pc_list = self.points.tolist()
        pc_msg = pc2.create_cloud_xyz32(header, pc_list)
        self.pc_pub.publish(pc_msg)
    
    def publish_cuboid_markers(self, stamp):
        marker_array = MarkerArray()
        marker_id = 0
        for cuboid in self.cuboids:
            lower = cuboid['lower']
            upper = cuboid['upper']
            dims = cuboid['dimensions_world']
            center = (lower + upper) / 2.0
            
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = stamp
            marker.ns = "free_cuboids"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(dims[0])
            marker.scale.y = float(dims[1])
            marker.scale.z = float(dims[2])
            marker.color.r = random.random()
            marker.color.g = random.random()
            marker.color.b = random.random()
            marker.color.a = 0.3
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

#############################################
# 6. Cuboid Connectivity + Graph Plotting
#############################################
def cuboids_touch_or_overlap(c1, c2, tol=0.0):
    """
    Returns True if cuboid c1 and c2 overlap or touch in all 3 dimensions
    (within an optional tolerance tol).
    c1, c2 are dicts with:
      'lower': np.array([x1, y1, z1])
      'upper': np.array([x2, y2, z2])
    """
    for i in range(3):
        if c1['upper'][i] < c2['lower'][i] - tol or c2['upper'][i] < c1['lower'][i] - tol:
            return False
    return True

def build_cuboid_graph(cuboids, tol=0.0):
    import networkx as nx
    G = nx.Graph()
    # Add nodes
    for i, cub in enumerate(cuboids):
        G.add_node(i, cuboid=cub)
    # Add edges if they overlap/touch
    for i in range(len(cuboids)):
        for j in range(i+1, len(cuboids)):
            if cuboids_touch_or_overlap(cuboids[i], cuboids[j], tol):
                G.add_edge(i, j)
    return G

def plot_and_save_cuboid_graph(G, out_file="cuboid_graph.png"):
    import networkx as nx
    import matplotlib.pyplot as plt
    
    pos = nx.spring_layout(G, seed=42)
    # Build labels
    labels = {}
    for node in G.nodes(data=True):
        idx = node[0]
        cub = node[1]['cuboid']
        lower = cub['lower'].round(1)
        dims = cub['dimensions_world'].round(1)
        labels[idx] = f"({tuple(lower)}, {tuple(dims)})"
    
    plt.figure(figsize=(8,6))
    nx.draw(G, pos, with_labels=False, node_color="steelblue", node_size=600, edge_color="black")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    plt.title("Cuboid Connectivity Graph")
    plt.axis('equal')
    
    # Save to file
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved cuboid connectivity graph to {out_file}")

#############################################
# 7. Main Routine
#############################################
def main(args=None):
    rclpy.init(args=args)
    
    # Load point cloud (Nx3 numpy array)
    pc_file = "/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/pointcloud/pointcloud_gq/point_cloud_gq.npy"
    points = np.load(pc_file)
    if points.dtype.names is not None:
        points = np.vstack([points[name] for name in ('x', 'y', 'z')]).T
    
    # Compute bounding box
    global_min = np.min(points, axis=0)
    global_max = np.max(points, axis=0)
    print("Point Cloud Bounding Box:")
    print("  Min:", global_min)
    print("  Max:", global_max)
    
    # Build occupancy grid
    resolution = 0.2
    occupancy = build_occupancy_grid(points, global_min, resolution)
    print(f"Occupancy grid shape: {occupancy.shape}, Occupied voxels: {occupancy.sum()}")
    
    # Expand obstacles (safety margin)
    safety_voxels = 2
    occupancy_expanded = expand_obstacles_3d(occupancy, safety_voxels)
    
    # =========== NEW: specify a fixed Z thickness for each cuboid =============
    max_z_thickness = 20  # <-- Adjust this as you see fit
    # Region growing -> free cuboids, limiting Z dimension to 'max_z_thickness'
    free_cuboids_blocks = region_growing_3d(occupancy_expanded, max_z_thickness=max_z_thickness)
    print("Number of free cuboids found:", len(free_cuboids_blocks))
    
    # Convert blocks to world coords
    cuboids = []
    for block in free_cuboids_blocks:
        print("[DEBUG] Block:", block)
        cub = block_to_world_cuboid(block, global_min, resolution)
        cuboids.append(cub)
    
    # Build + save the connectivity graph
    #G = build_cuboid_graph(cuboids, tol=0.0)
    #plot_and_save_cuboid_graph(G, out_file="my_cuboid_graph.png")
    
    # Publish to RViz
    node = RVizPublisher(points, cuboids, frame_id="map")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
