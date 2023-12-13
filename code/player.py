from vis_nav_game import Player, Action
import pygame
import cv2
import numpy as np
import PIL.Image as Image
from skimage.metrics import structural_similarity as compare_ssim
import math
import matplotlib.pyplot as plt
import time
import threading
import copy
import matplotlib.pyplot as plt  





# Camera intrinsic matrix
camera_matrix = np.array([[92., 0, 160.], [0, 92., 120.], [0, 0, 1]])


    
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        
        self.target_image = None # The target image
        self.captured_images = []
        self.most_similar_images = [None] * 4
        self.highest_similarities = [0] * 4
        self.most_similar_positions = [None] * 4
        self.lock = threading.Lock() 

        self.time_interval = 5  # Time interval in seconds to save images
        self.last_capture_time = time.time()
        self.time_flag = False
        

        self.fpv_counter = 0  # Add a counter for fpv frames

        self.camera_pos = np.array([0, 0]) # x, y

        self.camera_angle = 0  # Initial orientation of the camera
        self.rotate_flag = 0   # 0: no rotation, 1: rotate right, 2: rotate left
        self.fpv_frames = []  # List of fpv frames
        self.rotate_angle = 0  # Total rotation angle


        self.initial_frame = None  # Store the initial frame for 360-degree rotation
        self.frames_angle = 0
        self.rotate_360 = True

        self.move_flag = 0  # 0: no movement, 1: move forward, 2: move backward
        self.move_step = 0  # Total movement step

        depth_info_shape = (1, 1, 192, 640)
        self.depth_info = np.ones(depth_info_shape)

        self.positions = []  # To store camera positions
        plt.ion()  # Turn on interactive mode for live updates
        self.fig, self.ax = plt.subplots()  # Create a figure and axis for plotting
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)

        self.target_position = [None] * 4  # To store the target position

        super(KeyboardPlayerPyGame, self).__init__()

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    # Store the images and positions every 5 seconds
    def store_captured_image(self, image, camera_position):
        # Store the entire captured image
        current_time = time.time()
        time_elapsed = current_time - self.last_capture_time

        # Check if the time interval has elapsed
        if time_elapsed >= self.time_interval:
            # NOTE: We need to create a copy of the camera_position because it's a numpy array
            # otherwise, it will be overwritten in the next iteration

            # Create an actual copy of the camera_position
            camera_position_copy = np.copy(camera_position)

            # # Debugging: print the camera position before appending
            # print("Storing camera position:", camera_position_copy)

            self.captured_images.append((camera_position_copy, image))
            # print("self.captured_images: ", self.captured_images)
            self.last_capture_time = current_time   # Update the last capture time


    def compare_with_target_features(self, target_index, captured_images):
        # Load the target image
        target_image_path = f'target_temp_{target_index}.png'
        target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create()

        # Extract features from the target image
        target_keypoints, target_descriptors = orb.detectAndCompute(target_image, None)

        for i, (camera_position, captured_image) in enumerate(captured_images):
            # # Debugging: Print the index and camera position
            # print(f"Before Processing index {i}, camera_position: {camera_position}")

            # Extract features from the captured image
            keypoints, descriptors = orb.detectAndCompute(captured_image, None)

            # Create a BFMatcher (Brute Force Matcher) with Hamming distance
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Match the descriptors
            matches = bf.match(target_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            # # Debugging: Print the camera position again after processing
            # print(f"After Processed index {i}, camera_position: `{camera_position}`")


            # Calculate similarity
            if len(matches) > 0:
                with self.lock:
                    similarity = 1 - (matches[0].distance / len(target_descriptors))
                    if similarity > self.highest_similarities[target_index]:
                        self.highest_similarities[target_index] = similarity
                        self.most_similar_images[target_index] = captured_image
                        self.most_similar_positions[target_index] = camera_position

    def run_comparisons(self):
        threads = []

        # Assuming self.captured_images is a list of tuples [(camera_position, captured_image), ...]
        # We need to distribute this list equally among the threads
        num_cameras = len(self.captured_images)
        cameras_per_thread = num_cameras // 4

        for i in range(4):
            # Calculate the start and end indices for each subset of captured_images
            start_idx = i * cameras_per_thread
            end_idx = start_idx + cameras_per_thread

            # If it's the last thread, include any remaining cameras
            if i == 3:
                end_idx = num_cameras

            # Extract the subset for this thread
            captured_images_subset = self.captured_images[start_idx:end_idx]

            # Start the thread with the subset
            thread = threading.Thread(target=self.compare_with_target_features, args=(i, captured_images_subset,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()



        for i in range(4):
            if self.most_similar_images[i] is not None:
                # Prepare the text to be written
                text = f"Position: {self.most_similar_positions[i]}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_color = (255, 0, 0)  # Blue color in BGR
                line_type = 2

                # Copy the image to avoid modifying the original
                image_with_text = self.most_similar_images[i].copy()

                # Put the text on the image
                cv2.putText(image_with_text, text, (10, 30), font, font_scale, font_color, line_type)

                # Show the image
                window_name = f"Matched Image for Target {i}"
                cv2.imshow(window_name, image_with_text)

                self.target_position[i] = self.most_similar_positions[i]
                print(f"Target {i}: Highest similarity {self.highest_similarities[i]} at position {self.most_similar_positions[i]}")
            else:
                print(f"Target {i}: No similar image found.")

            
    def update_plot(self):
        # Append the new position
        self.positions.append((self.camera_pos[0], self.camera_pos[1]))

        # Clear only if there are no points to retain the color of previous points
        if not self.positions:
            self.ax.clear()

        # Plot each point except the last one and connect them with lines
        for i in range(len(self.positions) - 1):
            if not self.marker_colour:
                pos = self.positions[i]
                next_pos = self.positions[i + 1]
                self.ax.plot([pos[0], next_pos[0]], [pos[1], next_pos[1]], marker='o', color="red")  # Line connecting points in red
                self.ax.plot(pos[0], pos[1], marker='o', color="red")  # Plot each point in red

        # Check if there are at least two points to differentiate the start and end
        if len(self.positions) >= 2:
            # Plot the starting point in green
            self.ax.plot(self.positions[0][0], self.positions[0][1], marker='o', color="green")

        # Plot the last point with a black marker
        if self.marker_colour:
            # Plot the last point in blue if marker_colour is True
            self.ax.plot(self.positions[-1][0], self.positions[-1][1], marker='o', color="blue")
        else:
            # Plot the last point in black if marker_colour is False
            self.ax.plot(self.positions[-1][0], self.positions[-1][1], marker='o', color="black")
        
        # Plot the target position
        if self.target_position is not None and len(self.target_position) > 0:
            for i, target_pos in enumerate(self.target_position):
                if target_pos is not None:
                    # Plot the target position
                    self.ax.plot(target_pos[0], target_pos[1], marker='o', color="yellow")
                    self.ax.text(target_pos[0] + 0.1, target_pos[1] + 0.1, str(i), color="blue", fontsize=12)

                    # Find the corresponding camera position
                    for j, (camera_position, captured_image) in enumerate(self.captured_images):
                        # Use numpy.array_equal for comparison if they are numpy arrays
                        if np.array_equal(camera_position, target_pos):
                            # Plot all previous camera positions in purple
                            for k in range(j):
                                prev_camera_position = self.captured_images[k][0]
                                if k == 0:
                                    self.ax.plot(prev_camera_position[0], prev_camera_position[1], marker='x', color="purple")
                                else:
                                    next_camera_position = self.captured_images[k-1][0]
                                    self.ax.plot([prev_camera_position[0], next_camera_position[0]], 
                                                [prev_camera_position[1], next_camera_position[1]], 
                                                color="purple")
                            break


        self.ax.set_xlim(-200, 200)
        self.ax.set_ylim(-200, 200)
        # self.ax.set_xlim(-50, 50)
        # self.ax.set_ylim(-50, 50)
        plt.draw()
        plt.pause(0.001)

    
    def pre_exploration(self):
        self.marker_colour = False
        self.store_captured_images_flag = True

    def pre_navigation(self):
        self.marker_colour = True
        self.store_captured_images_flag = False


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
                        self.camera_pos = np.array([0, 0]) # x, y

                        self.camera_angle = 0  # Initial orientation of the camera
                        self.rotate_angle = 0  # Total rotation angle
                        self.time_flag = True
                        self.start_time = time.time()

                else:
                    self.show_target_images()
                    self.run_comparisons()


            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]

        # print("self.last_act: ", self.last_act)
        if self.last_act == Action.RIGHT:
            self.rotate_flag = 1
        elif self.last_act == Action.LEFT:
            self.rotate_flag = 2
        elif self.last_act == Action.FORWARD:
            self.move_flag = 1
        elif self.last_act == Action.BACKWARD:
            self.move_flag = 2
        else:
            if len(self.fpv_frames) > 0:
                print("length of self.fpv_frames = ", len(self.fpv_frames))
                if self.rotate_flag == 1:
                    self.rotate_angle = (self.frames_angle * len(self.fpv_frames)) % (2 * math.pi)
                elif self.rotate_flag == 2:
                    self.rotate_angle = - ((self.frames_angle * len(self.fpv_frames)) % (2 * math.pi))

                self.camera_angle += self.rotate_angle
                print("camera_angle = ", self.camera_angle)
            
            if self.move_step > 0:
                if self.move_flag == 1:
                    self.camera_pos[0] += self.move_step * math.cos(self.camera_angle)
                    self.camera_pos[1] += self.move_step * math.sin(self.camera_angle) 
                elif self.move_flag == 2:
                    self.camera_pos[0] -= self.move_step * math.cos(self.camera_angle) 
                    self.camera_pos[1] -= self.move_step * math.sin(self.camera_angle)
                print("camera position: ", self.camera_pos)

                # Update the plot after changing position
                self.update_plot()

            self.fpv_frames = []
            self.rotate_flag = 0

            self.move_step = 0
            self.move_flag = 0


        return self.last_act
    
    
    
    # Compare the similarity between two images
    def are_images_similar(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        score, _ = compare_ssim(gray1, gray2, full=True)
        return score > 0.95  # If the similarity score is greater than 0.95, we consider the two images are similar

    def record_fpv_frame(self):
        
        if self.rotate_flag != 0:
            # Only calculate onces
            if self.rotate_360:
                if self.initial_frame is None:
                    self.initial_frame = self.fpv
                else:
                    if self.are_images_similar(self.initial_frame, self.fpv) and len(self.fpv_frames) > 100:
                        print(f"Completed 360-degree rotation with {len(self.fpv_frames)} frames")
                        self.frames_angle = 2 * math.pi / len(self.fpv_frames)  # Calculate the angle between each frame
                        self.initial_frame = None  # Reset
                        self.rotate_360 = False
                        return
                    
            # Append the fpv frame to the list
            self.fpv_frames.append(self.fpv)
    
    def record_move_step(self):
        if self.move_flag != 0:
            self.move_step += 1
        

    def show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        self.target_image = concat_img

        selected_target = targets

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)


        for i, image in enumerate(selected_target):
            self.target_image_path = f'target_temp_{i}.png'  # Create a unique file name for each image
            cv2.imwrite(self.target_image_path, image)

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()



    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv


        self.record_fpv_frame()
            
        

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        if self.store_captured_images_flag: 
            self.store_captured_image(self.fpv, self.camera_pos)
        

        

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image


        
        pygame.display.set_caption("KeyboardPlayer:fpv")
        rgb = convert_opencv_img_to_pygame(fpv)

        self.screen.blit(rgb, (0, 0))

        # Display robot pose and time
        if self.time_flag:
            count_time = time.time()
            seconds = count_time - self.start_time
            seconds = int(seconds)
        else:
            seconds = 0
        

        font = pygame.font.Font(None, 36)
        text = font.render(f'Pose: {self.camera_pos}    Time: {seconds}', True, (0, 0, 255))
        self.screen.blit(text, (10, 10))
        
        pygame.display.update()


        self.fpv_counter += 1

        self.record_move_step()
        # print("move flag: ", self.move_flag)


if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())