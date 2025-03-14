from vis_nav_game import Player, Action
import math
import time
import numpy as np
import cv2
import pygame
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm
import random
import open3d as o3d
import concurrent.futures

# -------------------------------------------------------------------------
# Configuration Constants
# -------------------------------------------------------------------------
SCALE = 8
OFFSET = int(3 / (SCALE / 8))  # <-- Make sure OFFSET is defined
MIN_BASELINE = 1e-3  # Minimum baseline (in meters)

# Occupancy grid parameters (meters per cell)
OCC_RESOLUTION = 0.1

# Parameters for simple occupancy update (in meters)
CORRIDOR_OFFSET = 1.5  # Lateral offset from robot center for walls

# Occupancy codes (internal representation)
PATH_VAL = 1
WALL_VAL = 0
UNKNOWN_VAL = -1

NEW_RESOLUTION = 0.3  # Coarser resolution for merging occupancy
SAVE_FILENAME = "my_occupancy_map.npy"  # Output .npy file

# BGR color map for blocky images
COLOR_MAP = {
    UNKNOWN_VAL: np.array([255, 255, 255], dtype=np.uint8),  # white
    WALL_VAL: np.array([0, 0, 255], dtype=np.uint8),           # red
    PATH_VAL: np.array([0, 255, 255], dtype=np.uint8)          # yellow
}

# -------------------------------------------------------------------------
# Helper Functions for Mapping & Point Cloud Generation
# -------------------------------------------------------------------------
def normalize_angle(a):
    return a % (2 * math.pi)

def angle_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi

def images_are_similar(img1, img2, threshold=0.98):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = compare_ssim(gray1, gray2, full=True)
    return score >= threshold

def estimate_rotation_phase_correlation(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    center = (gray1.shape[1] // 2, gray1.shape[0] // 2)
    M = 40
    lp1 = cv2.logPolar(gray1, center, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    lp2 = cv2.logPolar(gray2, center, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    (shift, _) = cv2.phaseCorrelate(lp1, lp2)
    dtheta_deg = (shift[1] / lp1.shape[0]) * 360.0
    return math.radians(dtheta_deg)

def dynamic_world_to_map(x_world, y_world, origin, resolution):
    i = int((y_world - origin[1]) / resolution)
    j = int((x_world - origin[0]) / resolution)
    return (i, j)

def occupancy_to_rgb(occ):
    H, W = occ.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[occ == UNKNOWN_VAL] = [255, 255, 255]  # white
    rgb[occ == WALL_VAL] = [255, 0, 0]           # red
    rgb[occ == PATH_VAL] = [255, 255, 0]         # yellow
    return rgb

def convert_occ_to_blocky_image(occ, cell_size=10):
    H, W = occ.shape
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for key, color in COLOR_MAP.items():
        img[occ == key] = color
    return np.kron(img, np.ones((cell_size, cell_size, 1), dtype=np.uint8))

def merge_occupancy_grid(occ, new_res):
    block_size = int(new_res / OCC_RESOLUTION)
    H, W = occ.shape
    nH = H // block_size
    nW = W // block_size
    merged = np.full((nH, nW), UNKNOWN_VAL, dtype=int)
    for i in range(nH):
        for j in range(nW):
            block = occ[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            cpath = np.count_nonzero(block == PATH_VAL)
            cwall = np.count_nonzero(block == WALL_VAL)
            if cpath > cwall and cpath > 0:
                merged[i, j] = PATH_VAL
            elif cwall > 0:
                merged[i, j] = WALL_VAL
            else:
                merged[i, j] = UNKNOWN_VAL
    return merged

def dilate_path_cells(occ, kernel_size=7, iterations=2):
    mask = np.where(occ == PATH_VAL, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=iterations)
    out = occ.copy()
    newly_path = (dilated == 255) & (out == UNKNOWN_VAL)
    out[newly_path] = PATH_VAL
    return out

# --- Point Cloud Generation Functions ---
def get_disparity_map(left_img_arr, right_img_arr, edge_offset):
    gray_left = cv2.cvtColor(left_img_arr, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img_arr, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=16*6, blockSize=15)
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    disparity = disparity[edge_offset:-edge_offset, edge_offset:-edge_offset]
    return disparity

def get_distance_map(disparity_map, baseline, K):
    focal = K[0,0]
    H, W = disparity_map.shape
    distance_map = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            d = disparity_map[i, j]
            if d == 0:
                distance_map[i, j] = 0
            else:
                distance_map[i, j] = (focal * baseline) / d
    distances = np.zeros((H, W, 3), dtype=np.float32)
    cx = K[0,2]
    cy = K[1,2]
    for i in range(H):
        for j in range(W):
            Z = distance_map[i, j]
            X = (j - cx) * Z / focal
            Y = (i - cy) * Z / focal
            distances[i, j] = [X, Y, Z]
    return distances

def get_points(distance_map, image_arr):
    points = []
    H, W, _ = distance_map.shape
    for i in range(OFFSET, H - OFFSET):
        for j in range(OFFSET, W - OFFSET):
            X, Y, Z = distance_map[i, j]
            if Z == 0 or math.isnan(X) or math.isnan(Y) or math.isnan(Z):
                continue
            points.append([X, Y, Z])
    return points

def build_stereo_point_cloud(prev_img, curr_img, baseline, K):
    """Generate a point cloud from stereo images in the camera frame"""
    left_img_arr = prev_img.copy()
    right_img_arr = curr_img.copy()
    disparity = get_disparity_map(left_img_arr, right_img_arr, OFFSET)
    distances = get_distance_map(disparity, baseline, K)
    pts = get_points(distances, right_img_arr)
    return np.array(pts) if len(pts) > 0 else np.array([])

def get_transform_matrix(state):
    """Create a transformation matrix from robot state to world coordinates"""
    x, y, theta = state.flatten()
    # Create the transformation matrix (rotation + translation)
    transform = np.array([
        [math.cos(theta), -math.sin(theta), 0, x],
        [math.sin(theta), math.cos(theta), 0, y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return transform

def transform_points(points, transform_matrix):
    """Transform points from camera frame to world frame"""
    if len(points) == 0:
        return np.array([])
    
    # Convert to homogeneous coordinates
    homogeneous_points = np.ones((len(points), 4))
    homogeneous_points[:, :3] = points
    
    # Apply transformation
    transformed_points = homogeneous_points @ transform_matrix.T
    
    # Convert back from homogeneous coordinates
    return transformed_points[:, :3]

# Helper function for A* path planning
def astar_white(grid, start, goal):
    """
    A* path planning on a grid with white (255) as free cells.
    Returns a list of (i,j) indices forming the path from start to goal.
    """
    # Helpers
    def heuristic(a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def neighbors(node):
        # Define 8-connected neighborhood
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        result = []
        for d in dirs:
            neighbor = (node[0] + d[0], node[1] + d[1])
            if (0 <= neighbor[0] < grid.shape[0] and 
                0 <= neighbor[1] < grid.shape[1] and 
                grid[neighbor] == 255):
                result.append(neighbor)
        return result
    
    # Initialize data structures
    frontier = [(0, start)]  # Priority queue: (f_score, node)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    explored = set()
    
    while frontier:
        _, current = min(frontier)
        frontier.remove((f_score[current], current))
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        explored.add(current)
        
        for next_node in neighbors(current):
            if next_node in explored:
                continue
                
            tentative_g = g_score[current] + 1
            
            if next_node not in g_score or tentative_g < g_score[next_node]:
                came_from[next_node] = current
                g_score[next_node] = tentative_g
                f_score[next_node] = tentative_g + heuristic(next_node, goal)
                
                # Update the frontier
                frontier = [(f, n) for f, n in frontier if n != next_node]
                frontier.append((f_score[next_node], next_node))
        
    return None  # No path found

# -------------------------------------------------------------------------
# Main Player Class: EKF + Visual Measurement + Data Logging
# -------------------------------------------------------------------------
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        super().__init__()
        self.x = np.array([[0.0],[0.0],[0.0]])
        self.P = np.eye(3)*0.01
        self.u = np.array([[0.0],[0.0]])
        self.dt = 1.0
        self.prev_fpv = None
        self.prev_x = None
        self.K = None
        self.mapping_data = []  # List of (timestamp, state, FPV image)
        self.path = []          # List of (timestamp, X, Y, theta)
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.last_time = time.time()
        self.start_time = time.time()
        pygame.init()

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.x = np.array([[0.0],[0.0],[0.0]])
        self.P = np.eye(3)*0.01
        self.u = np.array([[0.0],[0.0]])
        self.prev_fpv = None
        self.prev_x = None
        self.mapping_data = []
        self.path = []
        self.start_time = time.time()
        self.last_time = time.time()
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def pre_exploration(self):
        self.K = self.get_camera_intrinsic_matrix()
        if self.K is None:
            self.K = np.array([[92., 0, 160.],
                               [0, 92., 120.],
                               [0, 0, 1]])
        print("Game started. We'll skip adding path entries if only turning in place.")

    def pre_navigation(self):
        pass

    def ekf_predict(self):
        now = time.time()
        self.dt = now - self.last_time
        self.last_time = now
        dx = self.dt * self.u[0,0]
        dth = self.dt * self.u[1,0]
        X, Y, th = self.x.flatten()
        Xp = X + dx * math.cos(th)
        Yp = Y + dx * math.sin(th)
        thp = normalize_angle(th + dth)
        self.x = np.array([[Xp],[Yp],[thp]])
        F = np.array([
            [1, 0, -dx*math.sin(th)],
            [0, 1,  dx*math.cos(th)],
            [0, 0, 1]
        ])
        Q = np.diag([0.01, 0.01, math.radians(1)**2])
        self.P = F @ self.P @ F.T + Q

    def ekf_update(self, z):
        H = np.array([[0, 0, 1]])
        R = np.array([[math.radians(2)**2]])
        y = np.array([[normalize_angle(z - self.x[2,0])]])
        S = H @ self.P @ H.T + R
        K_gain = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K_gain @ y
        self.x[2,0] = normalize_angle(self.x[2,0])
        self.P = (np.eye(3) - K_gain @ H) @ self.P

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                    if event.key == pygame.K_ESCAPE:
                        print("ESC pressed. Stopping updates.")
                        return Action.QUIT
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        speed = 0.5
        rot_speed = math.radians(2.5)
        u1 = 0.0
        u2 = 0.0
        if self.last_act & Action.FORWARD:
            u1 = speed / self.dt
        if self.last_act & Action.BACKWARD:
            u1 = -speed / self.dt
        if self.last_act & Action.RIGHT:
            u2 = -rot_speed / self.dt
        if self.last_act & Action.LEFT:
            u2 = rot_speed / self.dt
        self.u = np.array([[u1],[u2]])
        self.ekf_predict()
        return self.last_act

    def see(self, fpv):
        if fpv is None or fpv.ndim < 3:
            return
        self.fpv = fpv
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))
        img_rgb = cv2.cvtColor(fpv, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(np.flipud(np.rot90(img_rgb)))
        self.screen.blit(surf, (0, 0))
        font = pygame.font.SysFont(None, 24)
        pos_text = font.render(f'Pos:({self.x[0,0]:.2f},{self.x[1,0]:.2f})', True, (0, 0, 255))
        ang_text = font.render(f'Angle:{round(math.degrees(self.x[2,0]),1)}°', True, (0, 0, 255))
        self.screen.blit(pos_text, (10, 10))
        self.screen.blit(ang_text, (10, 30))
        pygame.display.update()

        t_stamp = time.time() - self.start_time
        self.mapping_data.append((t_stamp, self.x.copy(), self.fpv.copy()))
        print(f"Mapping data count: {len(self.mapping_data)}")
        if self.prev_fpv is None:
            self.prev_fpv = fpv.copy()
            self.prev_x = self.x.copy()
            print("First image captured; state initialized.")
        else:
            dtheta = estimate_rotation_phase_correlation(self.prev_fpv, fpv) * 0.25
            if dtheta is None or abs(dtheta) < math.radians(1):
                z = self.x[2,0]
            else:
                z = normalize_angle(self.prev_x[2,0] + dtheta)
                print(f"Visual measure dtheta = {math.degrees(dtheta):.1f}°")
            if images_are_similar(self.prev_fpv, fpv, threshold=0.99):
                z = self.x[2,0]
            self.ekf_update(z)
            self.prev_x = self.x.copy()
            self.prev_fpv = fpv.copy()
            print(f"Current state: {self.x.flatten()}")
        if (self.last_act & Action.FORWARD) or (self.last_act & Action.BACKWARD):
            self.path.append((t_stamp, self.x[0,0], self.x[1,0], self.x[2,0]))

    def show_target_images(self):
        tg = self.get_target_images()
        if tg is None or len(tg) <= 0:
            return
        hor1 = cv2.hconcat(tg[:2])
        hor2 = cv2.hconcat(tg[2:])
        cimg = cv2.vconcat([hor1, hor2])
        cv2.imshow("KeyboardPlayer:target_images", cimg)
        cv2.waitKey(1)

    def set_target_images(self, images):
        super().set_target_images(images)
        self.show_target_images()

    def compute_transformation(self, prev_state, curr_state):
        dtheta = angle_diff(curr_state[2], prev_state[2])
        R = np.array([[math.cos(dtheta), -math.sin(dtheta)],
                      [math.sin(dtheta),  math.cos(dtheta)]])
        t = np.array([[curr_state[0] - prev_state[0]],
                      [curr_state[1] - prev_state[1]]])
        T = np.eye(3)
        T[0:2, 0:2] = R
        T[0:2, 2:3] = t
        return T

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
if __name__=="__main__":
    from vis_nav_game import play
    player = KeyboardPlayerPyGame()
    play(the_player=player)

    print(f"Total mapping data entries: {len(player.mapping_data)}")

    # --- Build Occupancy Map from Recorded Path ---
    if len(player.path) == 0:
        print("No path data recorded (only turning?).")
    else:
        path = np.array(player.path)  # shape (N,4): [time, X, Y, theta]
        min_x, max_x = path[:,1].min(), path[:,1].max()
        min_y, max_y = path[:,2].min(), path[:,2].max()
        margin = 5.0
        origin = (min_x - margin, min_y - margin)
        map_w = (max_x - min_x) + 2 * margin
        map_h = (max_y - min_y) + 2 * margin
        grid_w = int(map_w / OCC_RESOLUTION)
        grid_h = int(map_h / OCC_RESOLUTION)
        
        occ_map = np.full((grid_h, grid_w), UNKNOWN_VAL, dtype=int)
        for (_, X, Y, theta) in path:
            i, j = dynamic_world_to_map(X, Y, origin, OCC_RESOLUTION)
            if 0 <= i < grid_h and 0 <= j < grid_w:
                occ_map[i, j] = PATH_VAL
            # left offset
            lx = X - CORRIDOR_OFFSET * math.sin(theta)
            ly = Y + CORRIDOR_OFFSET * math.cos(theta)
            iL, jL = dynamic_world_to_map(lx, ly, origin, OCC_RESOLUTION)
            if 0 <= iL < grid_h and 0 <= jL < grid_w:
                occ_map[iL, jL] = WALL_VAL
            # right offset
            rx = X + CORRIDOR_OFFSET * math.sin(theta)
            ry = Y - CORRIDOR_OFFSET * math.cos(theta)
            iR, jR = dynamic_world_to_map(rx, ry, origin, OCC_RESOLUTION)
            if 0 <= iR < grid_h and 0 <= jR < grid_w:
                occ_map[iR, jR] = WALL_VAL
        
        # Display fine occupancy grid
        fine_rgb = occupancy_to_rgb(occ_map)
        plt.figure(figsize=(10,10))
        plt.imshow(fine_rgb, origin='lower')
        plt.title("Fine Occupancy Grid")
        plt.show()

        # Merge to coarser grid and dilate path cells
        merged_occ = merge_occupancy_grid(occ_map, NEW_RESOLUTION)
        merged_dil = dilate_path_cells(merged_occ, kernel_size=7)
        np.save(SAVE_FILENAME, merged_dil)
        print(f"Saved occupancy map to {SAVE_FILENAME} with shape={merged_dil.shape}.")
        blocky = convert_occ_to_blocky_image(merged_dil, cell_size=20)
        plt.figure(figsize=(10,10))
        plt.imshow(cv2.cvtColor(blocky, cv2.COLOR_BGR2RGB))
        plt.title("Merged + Dilated Occupancy (Blocky View)")
        plt.axis('off')
        plt.show()

    # --- Point Cloud Generation and Visualization (UPDATED) ---
    if len(player.mapping_data) > 1:
        all_points = []
        path_points = []  # To store robot path points for visualization
        print("Building point cloud from mapping data...")
        
        # Add robot path points for visualization
        for t_stamp, state, _ in player.mapping_data:
            x, y, _ = state.flatten()
            path_points.append([x, y, 0])  # Z=0 for robot path
        
        # Process each pair of consecutive frames for stereo point cloud
        for i in range(1, len(player.mapping_data)):
            t_prev, state_prev, img_prev = player.mapping_data[i-1]
            t_curr, state_curr, img_curr = player.mapping_data[i]
            
            # Calculate baseline between frames (distance moved)
            baseline = np.linalg.norm(state_curr[:2].flatten() - state_prev[:2].flatten())
            if baseline < MIN_BASELINE:
                continue
            
            # IMPORTANT: We generate stereo point cloud using consecutive frames
            local_points = build_stereo_point_cloud(img_prev, img_curr, baseline, player.K)
            
            if len(local_points) > 0:
                # The important change: Create a transformation specific to this frame pair
                # We'll use the MIDPOINT between prev and curr states for better localization
                mid_state = np.zeros((3, 1))
                mid_state[0, 0] = (state_prev[0, 0] + state_curr[0, 0]) / 2  # X midpoint
                mid_state[1, 0] = (state_prev[1, 0] + state_curr[1, 0]) / 2  # Y midpoint
                mid_state[2, 0] = (state_prev[2, 0] + state_curr[2, 0]) / 2  # Theta midpoint
                
                # Create a camera-to-world transform based on robot position
                world_transform = np.eye(4)
                theta = mid_state[2, 0]
                
                # Rotation component (around Z axis)
                world_transform[0:3, 0:3] = np.array([
                    [math.cos(theta), -math.sin(theta), 0],
                    [math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1]
                ])
                
                # Translation component (X, Y, Z=0)
                world_transform[0, 3] = mid_state[0, 0]
                world_transform[1, 3] = mid_state[1, 0]
                
                # Transform local points to world frame
                homogeneous_points = np.ones((len(local_points), 4))
                homogeneous_points[:, :3] = local_points
                world_points = (homogeneous_points @ world_transform.T)[:, :3]
                
                # Filter out points that are too far from the robot path
                # This helps remove outliers from the stereo matching
                max_distance = 3.0  # Maximum distance from robot path in meters
                filtered_points = []
                for point in world_points:
                    x, y, z = point
                    dist_to_robot = np.linalg.norm(point[:2] - mid_state[:2].flatten())
                    if dist_to_robot < max_distance:
                        filtered_points.append(point)
                
                # Add to overall point cloud
                if filtered_points:
                    all_points.extend(filtered_points)
                    print(f"Frame {i}: Added {len(filtered_points)} points to world frame")
        
        print(f"Total points generated: {len(all_points)}")
        
        if len(all_points) > 0:
            # Create point cloud from environment points
            all_points = np.array(all_points)
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(all_points)
            
            # Add colors to points based on height
            colors = np.zeros((len(all_points), 3))
            z_values = all_points[:, 2]
            z_min, z_max = np.min(z_values), np.max(z_values)
            if z_max > z_min:
                normalized_z = (z_values - z_min) / (z_max - z_min)
                colors[:, 0] = 1 - normalized_z  # Red decreases with height
                colors[:, 1] = normalized_z      # Green increases with height
                pc.colors = o3d.utility.Vector3dVector(colors)
            
            # Create robot path line set for visualization
            path_points = np.array(path_points)
            lines = [[i, i+1] for i in range(len(path_points)-1)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(path_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])  # Red lines
            
            # Optional: statistical outlier removal for cleaner visualization
            pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Visualize point cloud and robot path together
            print("Displaying point cloud with robot path in Open3D.")
            o3d.visualization.draw_geometries([pc, line_set])
        else:
            print("No point cloud data generated from mapping data.")
    else:
        print("Not enough mapping data for point cloud generation.")

    # --- A* Path Planning on Merged Grid ---
    if len(player.path) == 0:
        print("No path data recorded.")
    else:
        path_arr = np.array(player.path)  # shape (N,4): [time, X, Y, theta]
        min_x, max_x = path_arr[:,1].min(), path_arr[:,1].max()
