from vis_nav_game import Player, Action

import math
import time
import numpy as np
import cv2
import pygame
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# -------------------------------------------------------------------------
# Configuration Constants
# -------------------------------------------------------------------------
OCC_RESOLUTION = 0.1     # Fine occupancy grid resolution (meters per cell)
CORRIDOR_OFFSET = 1.5    # Lateral offset from robot’s center (in meters) to mark as wall
MIN_BASELINE = 1e-3      # Not used here

# Occupancy codes (internal representation)
#   UNKNOWN_VAL (-1) → Unknown → display as white
#   WALL_VAL (0)     → Wall    → display as red
#   PATH_VAL (1)     → Path    → display as yellow
PATH_VAL = 1
WALL_VAL = 0
UNKNOWN_VAL = -1

# Coarser grid resolution for merging (meters per cell)
NEW_RESOLUTION = 0.3

# -------------------------------------------------------------------------
# Color Map for Display (BGR format)
# -------------------------------------------------------------------------
COLOR_MAP = {
    UNKNOWN_VAL: np.array([255, 255, 255], dtype=np.uint8),  # white
    WALL_VAL:    np.array([0, 0, 255],   dtype=np.uint8),      # red
    PATH_VAL:    np.array([0, 255, 255], dtype=np.uint8)       # yellow
}

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def normalize_angle(angle):
    return angle % (2 * math.pi)

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
    return i, j

def merge_occupancy_grid(occ, new_resolution):
    """
    Merge the fine occupancy grid (occ, at OCC_RESOLUTION) into a coarser grid at resolution new_resolution.
    For each block (block_size = new_resolution / OCC_RESOLUTION):
      - If number of PATH cells > number of WALL cells and at least one PATH exists → PATH.
      - Else if any WALL exists → WALL.
      - Otherwise → UNKNOWN.
    """
    block_size = int(new_resolution / OCC_RESOLUTION)
    H, W = occ.shape
    new_H = H // block_size
    new_W = W // block_size
    new_occ = np.full((new_H, new_W), UNKNOWN_VAL, dtype=int)
    for i in range(new_H):
        for j in range(new_W):
            block = occ[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            count_path = np.count_nonzero(block == PATH_VAL)
            count_wall = np.count_nonzero(block == WALL_VAL)
            if count_path > count_wall and count_path > 0:
                new_occ[i, j] = PATH_VAL
            elif count_wall > 0:
                new_occ[i, j] = WALL_VAL
            else:
                new_occ[i, j] = UNKNOWN_VAL
    return new_occ

def convert_occ_to_blocky_image(occ, cell_size=10):
    """
    Convert occupancy grid to a blocky BGR image.
    Each grid cell is repeated as a block of size cell_size x cell_size.
    """
    H, W = occ.shape
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for key, color in COLOR_MAP.items():
        img[occ == key] = color
    blocky_img = np.kron(img, np.ones((cell_size, cell_size, 1), dtype=np.uint8))
    return blocky_img

def astar_white(occ_img, start, goal):
    """
    A* search on a 2D occupancy image where white (255) indicates free.
    start and goal are grid indices (i, j).
    Returns a list of grid indices forming the path or None.
    """
    def heuristic(a, b):
        return math.hypot(b[0]-a[0], b[1]-a[1])
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                 (-1,-1),(-1,1),(1,-1),(1,1)]
    import heapq
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        close_set.add(current)
        for di, dj in neighbors:
            ni, nj = current[0]+di, current[1]+dj
            if not (0 <= ni < occ_img.shape[0] and 0 <= nj < occ_img.shape[1]):
                continue
            if occ_img[ni, nj] != 255:
                continue
            tentative_g = gscore[current] + heuristic(current, (ni, nj))
            if (ni, nj) in close_set and tentative_g >= gscore.get((ni, nj), float('inf')):
                continue
            if tentative_g < gscore.get((ni, nj), float('inf')) or (ni, nj) not in [x[1] for x in oheap]:
                came_from[(ni, nj)] = current
                gscore[(ni, nj)] = tentative_g
                fscore[(ni, nj)] = tentative_g + heuristic((ni, nj), goal)
                heapq.heappush(oheap, (fscore[(ni, nj)], (ni, nj)))
    return None

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
        print("Game started. Mapping will be performed in the navigation phase.")
        print("Only forward/backward moves are recorded for occupancy (turn-only poses are skipped).")

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
            [1, 0, -dx * math.sin(th)],
            [0, 1,  dx * math.cos(th)],
            [0, 0, 1]
        ])
        Q = np.diag([0.01, 0.01, math.radians(1)**2])
        self.P = F @ self.P @ F.T + Q

    def ekf_update(self, z):
        H = np.array([[0,0,1]])
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
        pos_text = font.render(f'Pos: ({self.x[0,0]:.2f}, {self.x[1,0]:.2f})', True, (0, 0, 255))
        ang_text = font.render(f'Angle: {round(math.degrees(self.x[2,0]),1)}°', True, (0, 0, 255))
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
                print(f"Visual measurement (scaled): dtheta = {math.degrees(dtheta):.1f}°")
            if images_are_similar(self.prev_fpv, fpv, threshold=0.99):
                z = self.x[2,0]
            self.ekf_update(z)
            self.prev_x = self.x.copy()
            self.prev_fpv = fpv.copy()
            print(f"Current state: {self.x.flatten()}")

        # Only record path if moving forward/backward (skip pure turns)
        if (self.last_act & Action.FORWARD) or (self.last_act & Action.BACKWARD):
            self.path.append((t_stamp, self.x[0,0], self.x[1,0], self.x[2,0]))

    def show_target_images(self):
        tg = self.get_target_images()
        if tg is None or len(tg) <= 0:
            return
        hor1 = cv2.hconcat(tg[:2])
        hor2 = cv2.hconcat(tg[2:])
        concat_img = cv2.vconcat([hor1, hor2])
        cv2.imshow("KeyboardPlayer:target_images", concat_img)
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
# Main Runner and Post-Mapping Processing
# -------------------------------------------------------------------------
if __name__=="__main__":
    from vis_nav_game import play
    player = KeyboardPlayerPyGame()
    play(the_player=player)

    print(f"Total mapping data entries: {len(player.mapping_data)}")

    # --- Build a Simple Occupancy Grid from the Recorded Path ---
    if len(player.path) == 0:
        print("No path data recorded (only turn actions detected).")
    else:
        path = np.array(player.path)  # shape (N,4): [time, X, Y, theta]
        # Compute bounding box with margin
        min_x, max_x = path[:,1].min(), path[:,1].max()
        min_y, max_y = path[:,2].min(), path[:,2].max()
        margin = 5.0
        origin = (min_x - margin, min_y - margin)
        map_w = (max_x - min_x) + 2 * margin
        map_h = (max_y - min_y) + 2 * margin
        grid_w = int(map_w / OCC_RESOLUTION)
        grid_h = int(map_h / OCC_RESOLUTION)

        # Initialize fine occupancy grid with UNKNOWN_VAL (-1)
        occ_map = np.full((grid_h, grid_w), UNKNOWN_VAL, dtype=int)

        # Fill the fine occupancy grid: mark robot positions as PATH and lateral offsets as WALL.
        for (t_stamp, X, Y, theta) in path:
            i, j = dynamic_world_to_map(X, Y, origin, OCC_RESOLUTION)
            if 0 <= i < grid_h and 0 <= j < grid_w:
                occ_map[i, j] = PATH_VAL
            # Left offset as wall
            lx = X - CORRIDOR_OFFSET * math.sin(theta)
            ly = Y + CORRIDOR_OFFSET * math.cos(theta)
            iL, jL = dynamic_world_to_map(lx, ly, origin, OCC_RESOLUTION)
            if 0 <= iL < grid_h and 0 <= jL < grid_w:
                occ_map[iL, jL] = WALL_VAL
            # Right offset as wall
            rx = X + CORRIDOR_OFFSET * math.sin(theta)
            ry = Y - CORRIDOR_OFFSET * math.cos(theta)
            iR, jR = dynamic_world_to_map(rx, ry, origin, OCC_RESOLUTION)
            if 0 <= iR < grid_h and 0 <= jR < grid_w:
                occ_map[iR, jR] = WALL_VAL

        # Display fine occupancy grid as an RGB image.
        def occupancy_to_rgb(occ):
            rgb = np.zeros((occ.shape[0], occ.shape[1], 3), dtype=np.uint8)
            rgb[occ == UNKNOWN_VAL] = [255, 255, 255]  # white
            rgb[occ == WALL_VAL] = [255, 0, 0]           # red
            rgb[occ == PATH_VAL] = [255, 255, 0]         # yellow
            return rgb

        fine_rgb = occupancy_to_rgb(occ_map)
        plt.figure(figsize=(10,10))
        plt.imshow(fine_rgb, origin='lower')
        plt.title("Fine Occupancy Grid (Skipping Turn-Only Poses)")
        plt.show()

        # Convert fine grid to blocky image (for visual inspection)
        def convert_occ_to_blocky_image(occ, cell_size=10):
            H, W = occ.shape
            img = np.zeros((H, W, 3), dtype=np.uint8)
            for key, color in COLOR_MAP.items():
                img[occ == key] = color
            blocky = np.kron(img, np.ones((cell_size, cell_size, 1), dtype=np.uint8))
            return blocky

        fine_blocky = convert_occ_to_blocky_image(occ_map, cell_size=10)
        plt.figure(figsize=(10,10))
        plt.imshow(cv2.cvtColor(fine_blocky, cv2.COLOR_BGR2RGB))
        plt.title("Blocky Occupancy Grid (Fine)")
        plt.axis('off')
        plt.show()

        # --- Merge Fine Grid into Coarser Grid for Planning ---
        merged_occ = merge_occupancy_grid(occ_map, NEW_RESOLUTION)
        # For planning, we consider PATH cells as free: set PATH_VAL to 255, others 0.
        planning_img = np.where(merged_occ == PATH_VAL, 255, 0).astype(np.uint8)

        # --- Post-Process Merged Grid: Dilate Free Cells to widen the corridor ---
        # Adjust the kernel size as needed; here we use a 3x3 kernel.
        kernel = np.ones((3, 3), np.uint8)
        planning_img_dilated = cv2.dilate(planning_img, kernel, iterations=1)

        # Display merged (coarse) grid as blocky image.
        merged_blocky = convert_occ_to_blocky_image(merged_occ, cell_size=20)
        plt.figure(figsize=(10,10))
        plt.imshow(cv2.cvtColor(merged_blocky, cv2.COLOR_BGR2RGB))
        plt.title("Merged (Coarse) Occupancy Grid")
        plt.axis('off')
        plt.show()

        # Display the dilated planning image.
        plt.figure(figsize=(10,10))
        plt.imshow(planning_img_dilated, origin='lower', cmap='gray')
        plt.title("Dilated Planning Image (Free Cells Widened)")
        plt.axis('off')
        plt.show()

        # --- A* Path Planning on the Dilated Planning Image ---
        free_cells = np.argwhere(planning_img_dilated == 255)
        if len(free_cells) < 2:
            print("Not enough free cells in merged grid for A* planning.")
        else:
            start_idx = tuple(random.choice(free_cells))
            goal_idx = tuple(random.choice(free_cells))
            print(f"Random start index: {start_idx}, Random goal index: {goal_idx}")
            plan = astar_white(planning_img_dilated, start_idx, goal_idx)
            if plan is None:
                print("A* could not find a path on the dilated merged grid.")
            else:
                plan_world = []
                for (i, j) in plan:
                    xw = j * NEW_RESOLUTION + origin[0] + NEW_RESOLUTION / 2.0
                    yw = i * NEW_RESOLUTION + origin[1] + NEW_RESOLUTION / 2.0
                    plan_world.append((xw, yw))
                plan_world = np.array(plan_world)
                print(f"A* found a path with length {len(plan_world)}.")
                plt.figure(figsize=(10,10))
                plt.imshow(planning_img_dilated, origin='lower', cmap='gray')
                plt.plot(plan_world[:,0], plan_world[:,1], 'b.-', label="A* Path")
                plt.title("Merged Occupancy Grid with A* Path")
                plt.legend()
                plt.show()

    # --- Display Path and State vs. Time ---
    if len(player.path) > 0:
        path = np.array(player.path)
        plt.figure(figsize=(10,8))
        plt.plot(path[:,1], path[:,2], '-o', markersize=3, label='Path')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('2D Path Followed (Skipping Turn-Only Poses)')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

        fig, ax1 = plt.subplots(figsize=(10,4))
        ax1.plot(path[:,0], path[:,1], '-o', markersize=3, label='X')
        ax1.plot(path[:,0], path[:,2], '-o', markersize=3, label='Y')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (m)')
        ax1.set_title('Position vs. Time')
        ax1.legend()
        ax1.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10,4))
        plt.plot(path[:,0], np.degrees(path[:,3]), '-o', markersize=3, color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Orientation (°)')
        plt.title('Orientation vs. Time')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No path data recorded.")
