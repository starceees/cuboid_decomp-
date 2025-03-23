#!/usr/bin/env python3
import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import sensor_msgs_py.point_cloud2 as pc2
import random
import heapq

# Additional imports for progress and graph
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

#############################################
# 1. Obstacle Expansion (3D Safety Margin)
#############################################
def expand_obstacles_3d(occupancy, safety_voxels=2):
    """
    Expands obstacles in a 3D occupancy grid by 'safety_voxels' using morphological dilation.
    occupancy: 3D numpy array (1=obstacle, 0=free).
    Returns a new 3D array with obstacles expanded.
    """
    from scipy.ndimage import generate_binary_structure, binary_dilation
    structure = generate_binary_structure(rank=3, connectivity=1)  # 6-connected 3D structure
    expanded = binary_dilation(occupancy.astype(bool), structure=structure, iterations=safety_voxels)
    return expanded.astype(np.uint8)

#############################################
# 2. Build 3D Occupancy Grid from Point Cloud
#############################################
def build_occupancy_grid(points, global_min, resolution):
    """
    Given Nx3 point cloud and resolution, builds a 3D occupancy grid (1=occupied, 0=free).
    """
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
# 3. Region Growing for Free Cuboids
#############################################
def region_growing_3d(occupancy, max_z_thickness=5):
    """
    Greedily grows free cuboids in X/Y while capping the thickness in Z to 'max_z_thickness'.
    Returns a list of cuboid blocks (grid indices).
    """
    nx, ny, nz = occupancy.shape
    visited = np.zeros_like(occupancy, dtype=bool)
    cuboids = []
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if occupancy[i, j, k] == 0 and not visited[i, j, k]:
                    i_min, i_max = i, i
                    j_min, j_max = j, j
                    k_min, k_max = k, k
                    changed = True
                    while changed:
                        changed = False
                        # Expand +x
                        if i_max < nx - 1:
                            candidate = occupancy[i_max+1, j_min:j_max+1, k_min:k_max+1]
                            if np.all(candidate == 0):
                                i_max += 1
                                changed = True
                        # Expand +y
                        if j_max < ny - 1:
                            candidate = occupancy[i_min:i_max+1, j_max+1, k_min:k_max+1]
                            if np.all(candidate == 0):
                                j_max += 1
                                changed = True
                        # Expand +z but limit thickness
                        current_z = k_max - k_min + 1
                        if current_z < max_z_thickness and k_max < nz - 1:
                            candidate = occupancy[i_min:i_max+1, j_min:j_max+1, k_max+1]
                            if np.all(candidate == 0):
                                k_max += 1
                                changed = True
                    visited[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1] = True
                    cuboids.append({
                        'min_idx': np.array([i_min, j_min, k_min]),
                        'dimensions': np.array([i_max - i_min + 1,
                                                j_max - j_min + 1,
                                                k_max - k_min + 1])
                    })
    return cuboids

#############################################
# 4. Convert Grid Cuboid to World Coordinates
#############################################
def block_to_world_cuboid(block, global_min, resolution):
    """
    Convert a block (min_idx + dimensions) to a dict with 'lower','upper','dimensions_world'
    in world coordinates.
    """
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
# 5. Strict "Line-of-Sight" Checking
#############################################
def line_of_sight_in_cuboids(cub1, cub2, step=0.05):
    """
    Returns True if the entire line segment from cub1 center to cub2 center
    is contained in the union of cub1 and cub2.
    
    step: fraction of the segment to step each time (e.g. 0.05).
    """
    c1 = (cub1['lower'] + cub1['upper']) / 2.0
    c2 = (cub2['lower'] + cub2['upper']) / 2.0
    vec = c2 - c1
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return True  # practically the same center
    steps = int(np.ceil(1.0 / step))
    for i in range(steps + 1):
        t = i * step
        if t > 1.0:
            t = 1.0
        point = c1 + t * vec
        if not point_in_union_of_two_cuboids(point, cub1, cub2):
            return False
    return True

def point_in_union_of_two_cuboids(pt, cub1, cub2):
    return point_in_cuboid(pt, cub1) or point_in_cuboid(pt, cub2)

def point_in_cuboid(pt, cub):
    return np.all(pt >= cub['lower']) and np.all(pt <= cub['upper'])

#############################################
# 6. Build Cuboids with Connectivity (Graph on the Fly)
#############################################
def build_cuboids_with_connectivity(free_cuboids_blocks, global_min, resolution, step=0.05):
    """
    Convert each free cuboid block to world coords.
    Then check connectivity by:
      1) Overlap/touch in bounding sense
      2) The entire line from center1 to center2 is in the union (line_of_sight_in_cuboids).
    """
    cuboids = []
    for block in tqdm(free_cuboids_blocks, desc="Building cuboid connectivity"):
        cub = block_to_world_cuboid(block, global_min, resolution)
        cub['neighbors'] = []
        for idx, existing in enumerate(cuboids):
            # Basic bounding check
            if cuboids_touch_or_overlap(cub, existing, tol=0.0):
                # Strict line-of-sight check
                if line_of_sight_in_cuboids(cub, existing, step=step):
                    cub['neighbors'].append(idx)
                    existing['neighbors'].append(len(cuboids))
        cuboids.append(cub)
    return cuboids

def cuboids_touch_or_overlap(c1, c2, tol=0.0):
    """
    Returns True if c1 and c2 overlap/touch in all 3 dims (within optional tol).
    """
    for i in range(3):
        if c1['upper'][i] < c2['lower'][i] - tol or c2['upper'][i] < c1['lower'][i] - tol:
            return False
    return True

#############################################
# 7. A* Over Cuboids
#############################################
def astar_on_cuboids(cuboids, start_idx, goal_idx):
    """
    Runs A* where each cuboid has a 'neighbors' list and we use Eucl dist between centers.
    """
    def center(cub):
        return (cub['lower'] + cub['upper']) / 2.0
    def heuristic(i, j):
        return np.linalg.norm(center(cuboids[i]) - center(cuboids[j]))
    
    open_set = []
    best_cost = {start_idx: 0}
    visited = set()
    heapq.heappush(open_set, (heuristic(start_idx, goal_idx), 0, start_idx, [start_idx]))
    
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current == goal_idx:
            return path
        if current in visited:
            continue
        visited.add(current)
        for nb in cuboids[current]['neighbors']:
            if nb in visited:
                continue
            cost_new = g + heuristic(current, nb)
            if nb not in best_cost or cost_new < best_cost[nb]:
                best_cost[nb] = cost_new
                f_new = cost_new + heuristic(nb, goal_idx)
                heapq.heappush(open_set, (f_new, cost_new, nb, path + [nb]))
    return None

#############################################
# 8. ROS2 Node for Publishing
#############################################
class RVizPublisher(Node):
    def __init__(self, points, cuboids, path_indices, frame_id="map"):
        """
        :param points: Nx3 numpy array
        :param cuboids: list of dicts with 'lower','upper','neighbors'
        :param path_indices: list of indices (cuboids) forming the path
        """
        super().__init__('rviz_los_planner')
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pc_pub = self.create_publisher(PointCloud2, '/point_cloud', qos)
        self.marker_pub_all = self.create_publisher(MarkerArray, '/free_cuboids', qos)
        self.marker_pub_path = self.create_publisher(MarkerArray, '/path_cuboids', qos)
        self.path_pub = self.create_publisher(Path, '/planned_path', qos)
        
        self.points = points
        self.cuboids = cuboids
        self.path_indices = path_indices
        self.frame_id = frame_id
        
        # Build the actual 3D path coords
        self.path_coords = []
        for idx in path_indices:
            c = cuboids[idx]
            center = (c['lower'] + c['upper']) / 2.0
            self.path_coords.append(center)
        
        self.timer = self.create_timer(1.0, self.publish_all)
        self.get_logger().info("LOS Planner Node initialized.")
    
    def publish_all(self):
        now = self.get_clock().now().to_msg()
        self.publish_point_cloud(now)
        self.publish_all_cuboids(now)
        self.publish_path_cuboids(now)
        self.publish_path(now)
        self.get_logger().info("Published all data (cloud, cuboids, path).")
    
    def publish_point_cloud(self, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = self.frame_id
        pc_list = self.points.tolist()
        pc_msg = pc2.create_cloud_xyz32(header, pc_list)
        self.pc_pub.publish(pc_msg)
    
    def publish_all_cuboids(self, stamp):
        marker_array = MarkerArray()
        mid = 0
        for cub in self.cuboids:
            lower = cub['lower']
            upper = cub['upper']
            dims = cub['dimensions_world']
            center = (lower + upper) / 2.0
            
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = "all_cuboids"
            m.id = mid
            mid += 1
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(center[0])
            m.pose.position.y = float(center[1])
            m.pose.position.z = float(center[2])
            m.pose.orientation.w = 1.0
            m.scale.x = float(dims[0])
            m.scale.y = float(dims[1])
            m.scale.z = float(dims[2])
            m.color.r = 0.2
            m.color.g = 0.8
            m.color.b = 0.2
            m.color.a = 0.3
            marker_array.markers.append(m)
        self.marker_pub_all.publish(marker_array)
    
    def publish_path_cuboids(self, stamp):
        marker_array = MarkerArray()
        mid = 0
        for idx in self.path_indices:
            cub = self.cuboids[idx]
            lower = cub['lower']
            upper = cub['upper']
            dims = cub['dimensions_world']
            center = (lower + upper) / 2.0
            
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = "path_cuboids"
            m.id = mid
            mid += 1
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(center[0])
            m.pose.position.y = float(center[1])
            m.pose.position.z = float(center[2])
            m.pose.orientation.w = 1.0
            m.scale.x = float(dims[0])
            m.scale.y = float(dims[1])
            m.scale.z = float(dims[2])
            m.color.r = 0.9
            m.color.g = 0.1
            m.color.b = 0.1
            m.color.a = 0.8
            marker_array.markers.append(m)
        self.marker_pub_path.publish(marker_array)
    
    def publish_path(self, stamp):
        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = self.frame_id
        for pt in self.path_coords:
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = float(pt[0])
            pose.pose.position.y = float(pt[1])
            pose.pose.position.z = float(pt[2])
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

#############################################
# 9. Main Routine
#############################################
def main(args=None):
    rclpy.init(args=args)
    
    # Load your point cloud
    pc_file = "/home/raghuram/ARPL/cuboid_decomp/cuboid_decomp-/pointcloud/pointcloud_gq/point_cloud_gq.npy"  # <-- update as needed
    points = np.load(pc_file)
    if points.dtype.names is not None:
        points = np.vstack([points[name] for name in ('x','y','z')]).T
    
    # Compute bounding box
    global_min = np.min(points, axis=0)
    global_max = np.max(points, axis=0)
    print("Point Cloud Bounding Box:")
    print("  Min:", global_min)
    print("  Max:", global_max)
    
    # Build occupancy
    resolution = 0.2
    occupancy = build_occupancy_grid(points, global_min, resolution)
    print("Occupancy shape:", occupancy.shape, "occupied count:", occupancy.sum())
    
    # Expand obstacles
    safety_voxels = 2
    occupancy_expanded = expand_obstacles_3d(occupancy, safety_voxels)
    
    # Region growing
    max_z_thickness = 20
    blocks = region_growing_3d(occupancy_expanded, max_z_thickness)
    print("Found", len(blocks), "free cuboids.")
    
    # Build connectivity with line-of-sight checks
    # step=0.05 => about 20 samples along each line-of-sight
    cuboids = build_cuboids_with_connectivity(blocks, global_min, resolution, step=0.05)
    
    # Randomly pick a start & goal
    start_idx = random.randint(0, len(cuboids)-1)
    goal_idx = random.randint(0, len(cuboids)-1)
    while goal_idx == start_idx:
        goal_idx = random.randint(0, len(cuboids)-1)
    print("start:", start_idx, "goal:", goal_idx)
    
    # Plan path
    path = astar_on_cuboids(cuboids, start_idx, goal_idx)
    if path is None:
        print("No path found between start and goal!")
        path = []
    else:
        print("Path:", path)
    
    # Publish results
    node = RVizPublisher(points, cuboids, path, frame_id="map")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
