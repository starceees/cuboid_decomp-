from vis_nav_game import Player, Action
import math, time, random
import numpy as np
import cv2
import pygame
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm

# -------------------------------------------------------------------------
# Configuration Constants
# -------------------------------------------------------------------------
OCC_RESOLUTION = 0.1
CORRIDOR_OFFSET = 1.5
MIN_BASELINE = 1e-3

PATH_VAL = 1
WALL_VAL = 0
UNKNOWN_VAL = -1

NEW_RESOLUTION = 0.3  # coarser grid resolution

COLOR_MAP = {
    UNKNOWN_VAL: np.array([255, 255, 255], dtype=np.uint8),
    WALL_VAL:    np.array([0, 0, 255], dtype=np.uint8),
    PATH_VAL:    np.array([0, 255, 255], dtype=np.uint8)
}

def normalize_angle(a): return a % (2*math.pi)
def angle_diff(a,b): return (a - b + math.pi) % (2*math.pi) - math.pi

def images_are_similar(img1, img2, threshold=0.98):
    import skimage
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = compare_ssim(gray1, gray2, full=True)
    return score >= threshold

def estimate_rotation_phase_correlation(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    center = (gray1.shape[1]//2, gray1.shape[0]//2)
    M = 40
    lp1 = cv2.logPolar(gray1, center, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    lp2 = cv2.logPolar(gray2, center, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    (shift, _) = cv2.phaseCorrelate(lp1, lp2)
    dtheta_deg = (shift[1]/lp1.shape[0])*360.0
    return math.radians(dtheta_deg)

def dynamic_world_to_map(x_world, y_world, origin, resolution):
    i = int((y_world - origin[1]) / resolution)
    j = int((x_world - origin[0]) / resolution)
    return (i, j)

def merge_occupancy_grid(occ, new_res):
    block_size = int(new_res/OCC_RESOLUTION)
    H,W = occ.shape
    nH = H//block_size
    nW = W//block_size
    new_occ = np.full((nH,nW), UNKNOWN_VAL, dtype=int)
    for i in range(nH):
        for j in range(nW):
            block = occ[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            cpath = np.count_nonzero(block==PATH_VAL)
            cwall = np.count_nonzero(block==WALL_VAL)
            if cpath>cwall and cpath>0:
                new_occ[i,j] = PATH_VAL
            elif cwall>0:
                new_occ[i,j] = WALL_VAL
            else:
                new_occ[i,j] = UNKNOWN_VAL
    return new_occ

def convert_occ_to_blocky_image(occ, cell_size=10):
    H,W = occ.shape
    img = np.zeros((H,W,3), dtype=np.uint8)
    for key,color in COLOR_MAP.items():
        img[occ==key] = color
    return np.kron(img, np.ones((cell_size,cell_size,1), dtype=np.uint8))

def astar_white(occ_img, start, goal):
    def heuristic(a,b): return math.hypot(b[0]-a[0], b[1]-a[1])
    neighbors=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    import heapq
    close_set=set()
    came={}
    gscore={start:0}
    fscore={start:heuristic(start,goal)}
    oheap=[]
    heapq.heappush(oheap,(fscore[start],start))
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current==goal:
            path=[]
            while current in came:
                path.append(current)
                current=came[current]
            path.append(start)
            path.reverse()
            return path
        close_set.add(current)
        for di,dj in neighbors:
            ni,nj=current[0]+di, current[1]+dj
            if not(0<=ni<occ_img.shape[0] and 0<=nj<occ_img.shape[1]):
                continue
            if occ_img[ni,nj]!=255:
                continue
            tg = gscore[current]+heuristic(current,(ni,nj))
            if (ni,nj) in close_set and tg>=gscore.get((ni,nj),float('inf')):
                continue
            if tg<gscore.get((ni,nj),float('inf')) or (ni,nj) not in [xx[1] for xx in oheap]:
                came[(ni,nj)]=current
                gscore[(ni,nj)]=tg
                fscore[(ni,nj)]=tg+heuristic((ni,nj),goal)
                heapq.heappush(oheap,(fscore[(ni,nj)],(ni,nj)))
    return None

# Optionally apply morphological dilation to widen free space
def dilate_image(img, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
    return cv2.dilate(img, kernel, iterations=1)

# -------------------------------------------------------------------------
# The Player
# -------------------------------------------------------------------------
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        super().__init__()
        self.x = np.array([[0.0],[0.0],[0.0]])
        self.P = np.eye(3)*0.01
        self.u = np.array([[0.0],[0.0]])
        self.dt = 1.0
        self.prev_fpv = None
        self.prev_x   = None
        self.K = None
        self.mapping_data = []
        self.path = []
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.last_time = time.time()
        self.start_time= time.time()
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
        self.start_time= time.time()
        self.last_time= time.time()
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
        print("Game started. We skip adding path entries when only turning in place.")

    def pre_navigation(self):
        pass

    def ekf_predict(self):
        now = time.time()
        self.dt = now - self.last_time
        self.last_time = now
        dx = self.dt * self.u[0,0]
        dth= self.dt * self.u[1,0]
        X, Y, th = self.x.flatten()
        Xp = X + dx*math.cos(th)
        Yp = Y + dx*math.sin(th)
        thp= normalize_angle(th + dth)
        self.x = np.array([[Xp],[Yp],[thp]])
        F = np.array([
            [1, 0, -dx*math.sin(th)],
            [0, 1,  dx*math.cos(th)],
            [0, 0, 1]
        ])
        Q = np.diag([0.01,0.01,math.radians(1)**2])
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
            u1 = speed/self.dt
        if self.last_act & Action.BACKWARD:
            u1 = -speed/self.dt
        if self.last_act & Action.RIGHT:
            u2 = -rot_speed/self.dt
        if self.last_act & Action.LEFT:
            u2 =  rot_speed/self.dt
        self.u = np.array([[u1],[u2]])
        self.ekf_predict()
        return self.last_act

    def see(self, fpv):
        if fpv is None or fpv.ndim < 3:
            return
        self.fpv = fpv
        if self.screen is None:
            h,w,_=fpv.shape
            self.screen = pygame.display.set_mode((w,h))
        img_rgb=cv2.cvtColor(fpv,cv2.COLOR_BGR2RGB)
        surf=pygame.surfarray.make_surface(np.flipud(np.rot90(img_rgb)))
        self.screen.blit(surf,(0,0))
        font=pygame.font.SysFont(None,24)
        pos_text=font.render(f'Pos:({self.x[0,0]:.2f},{self.x[1,0]:.2f})',True,(0,0,255))
        ang_text=font.render(f'Angle:{round(math.degrees(self.x[2,0]),1)}°',True,(0,0,255))
        self.screen.blit(pos_text,(10,10))
        self.screen.blit(ang_text,(10,30))
        pygame.display.update()

        t_stamp = time.time()-self.start_time
        self.mapping_data.append((t_stamp, self.x.copy(), fpv.copy()))
        print(f"Mapping data count: {len(self.mapping_data)}")

        if self.prev_fpv is None:
            self.prev_fpv=fpv.copy()
            self.prev_x=self.x.copy()
            print("First image captured; state init.")
        else:
            dtheta=estimate_rotation_phase_correlation(self.prev_fpv,fpv)*0.25
            if dtheta is None or abs(dtheta)<math.radians(1):
                z=self.x[2,0]
            else:
                z=normalize_angle(self.prev_x[2,0]+dtheta)
                print(f"Visual measure (scaled) dtheta={math.degrees(dtheta):.1f}°")
            if images_are_similar(self.prev_fpv,fpv,threshold=0.99):
                z=self.x[2,0]
            self.ekf_update(z)
            self.prev_x=self.x.copy()
            self.prev_fpv=fpv.copy()

        # only record path if moving forward/backward
        if (self.last_act & Action.FORWARD) or (self.last_act & Action.BACKWARD):
            self.path.append((t_stamp, self.x[0,0], self.x[1,0], self.x[2,0]))

    def show_target_images(self):
        tg=self.get_target_images()
        if tg is None or len(tg)<=0:
            return
        hor1=cv2.hconcat(tg[:2])
        hor2=cv2.hconcat(tg[2:])
        cimg=cv2.vconcat([hor1,hor2])
        cv2.imshow("KeyboardPlayer:target_images",cimg)
        cv2.waitKey(1)

    def set_target_images(self,images):
        super().set_target_images(images)
        self.show_target_images()

    def compute_transformation(self, prev_state, curr_state):
        dtheta=angle_diff(curr_state[2],prev_state[2])
        R=np.array([[math.cos(dtheta),-math.sin(dtheta)],
                    [math.sin(dtheta), math.cos(dtheta)]])
        t=np.array([[curr_state[0]-prev_state[0]],
                    [curr_state[1]-prev_state[1]]])
        T=np.eye(3)
        T[0:2,0:2]=R
        T[0:2,2:3]=t
        return T

# -------------------------------------------------------------------------
# Main + Post-Processing
# -------------------------------------------------------------------------
if __name__=="__main__":
    from vis_nav_game import play
    player=KeyboardPlayerPyGame()
    play(the_player=player)

    print(f"Total mapping data entries: {len(player.mapping_data)}")

    if len(player.path)==0:
        print("No path data recorded (only turning?).")
    else:
        # 1) Build Fine Grid
        path = np.array(player.path)
        min_x, max_x = path[:,1].min(), path[:,1].max()
        min_y, max_y = path[:,2].min(), path[:,2].max()
        margin=5.0
        origin=(min_x-margin, min_y-margin)
        map_w=(max_x-min_x)+2*margin
        map_h=(max_y-min_y)+2*margin
        grid_w=int(map_w/OCC_RESOLUTION)
        grid_h=int(map_h/OCC_RESOLUTION)

        occ_map=np.full((grid_h,grid_w), UNKNOWN_VAL,dtype=int)
        for (_,X,Y,theta) in path:
            i,j=dynamic_world_to_map(X,Y,origin,OCC_RESOLUTION)
            if 0<=i<grid_h and 0<=j<grid_w:
                occ_map[i,j]=PATH_VAL
            # left
            lx=X - CORRIDOR_OFFSET*math.sin(theta)
            ly=Y + CORRIDOR_OFFSET*math.cos(theta)
            iL,jL=dynamic_world_to_map(lx,ly,origin,OCC_RESOLUTION)
            if 0<=iL<grid_h and 0<=jL<grid_w:
                occ_map[iL,jL]=WALL_VAL
            # right
            rx=X + CORRIDOR_OFFSET*math.sin(theta)
            ry=Y - CORRIDOR_OFFSET*math.cos(theta)
            iR,jR=dynamic_world_to_map(rx,ry,origin,OCC_RESOLUTION)
            if 0<=iR<grid_h and 0<=jR<grid_w:
                occ_map[iR,jR]=WALL_VAL

        # 2) Display Fine Grid
        def occupancy_to_rgb(occ):
            rgb = np.zeros((occ.shape[0],occ.shape[1],3),dtype=np.uint8)
            rgb[occ==UNKNOWN_VAL] = [255,255,255]
            rgb[occ==WALL_VAL]    = [255,0,0]
            rgb[occ==PATH_VAL]    = [255,255,0]
            return rgb

        fine_rgb=occupancy_to_rgb(occ_map)
        plt.figure(figsize=(10,10))
        plt.imshow(fine_rgb,origin='lower')
        plt.title("Fine Occupancy Grid")
        plt.show()

        # 3) Merge Fine Grid -> Coarser
        merged_occ=merge_occupancy_grid(occ_map, NEW_RESOLUTION)

        # Convert PATH -> white(255), others -> black(0)
        plan_img=np.where(merged_occ==PATH_VAL,255,0).astype(np.uint8)

        # 4) Optionally DILATE plan_img to widen corridor
        plan_img_dilated = dilate_image(plan_img, kernel_size=3)  # adjust kernel_size if needed

        # Show merged/dilated in grayscale
        plt.figure(figsize=(8,8))
        plt.imshow(plan_img_dilated, origin='lower', cmap='gray')
        plt.title("Merged + Dilated Grid (white=free, black=blocked)")
        plt.show()

        # 5) A* on plan_img_dilated
        free_cells = np.argwhere(plan_img_dilated==255)
        if len(free_cells)<2:
            print("Not enough free cells after dilation.")
        else:
            import random
            sidx = tuple(random.choice(free_cells))
            gidx = tuple(random.choice(free_cells))
            print(f"Random start={sidx}, goal={gidx}")

            path_ij = astar_white(plan_img_dilated, sidx, gidx)
            if path_ij is None:
                print("A* could not find path after dilation.")
            else:
                # Convert coarse grid cell (i,j) back to real-world coords
                # The merged grid has shape (nH, nW). Each cell is size NEW_RESOLUTION.
                # i->y, j->x, origin is the same, plus offset of half cell.
                path_xy = []
                for (i,j) in path_ij:
                    xw = j*NEW_RESOLUTION + origin[0] + NEW_RESOLUTION/2.0
                    yw = i*NEW_RESOLUTION + origin[1] + NEW_RESOLUTION/2.0
                    path_xy.append((xw,yw))
                path_xy=np.array(path_xy)

                # Plot the path in the same axes
                fig, ax=plt.subplots(figsize=(8,8))
                ax.imshow(plan_img_dilated, origin='lower', cmap='gray',
                          extent=[origin[0], origin[0]+(merged_occ.shape[1]*NEW_RESOLUTION),
                                  origin[1], origin[1]+(merged_occ.shape[0]*NEW_RESOLUTION)])
                ax.plot(path_xy[:,0], path_xy[:,1], 'b.-', label="A* Path")
                ax.set_title("Merged Occupancy Grid with A* Path")
                ax.legend()
                plt.show()

    # 6) Basic pose/time plots
    if len(player.path)>0:
        path_arr=np.array(player.path)
        plt.figure(figsize=(10,8))
        plt.plot(path_arr[:,1], path_arr[:,2], '-o', markersize=3, label='Path')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('2D Path Followed (Skipping Turn-Only Poses)')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

        fig, ax=plt.subplots(figsize=(10,4))
        ax.plot(path_arr[:,0], path_arr[:,1], '-o', markersize=3, label='X')
        ax.plot(path_arr[:,0], path_arr[:,2], '-o', markersize=3, label='Y')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')
        ax.set_title('Position vs. Time')
        ax.legend()
        ax.grid(True)
        plt.show()

        plt.figure(figsize=(10,4))
        plt.plot(path_arr[:,0], np.degrees(path_arr[:,3]), '-o', markersize=3, color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Orientation (°)')
        plt.title('Orientation vs. Time')
        plt.grid(True)
        plt.show()
    else:
        print("No path data recorded.")
