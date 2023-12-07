from vis_nav_game import Player, Action
import pygame
import cv2
import time
import numpy as np
import math


class KeyboardPlayerPyGame(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.target_image = None # The target image
        self.captured_images = []

        self.distance_count = 0  # To track the distance moved
        self.move_distance = 0
        self.move_step = 1

        self.current_angle = 0  # Add this line to store the current angle
        self.turning_angle = 2.4  # The angle increment for each key press

        # self.position = np.array([0, 0])
        self.position = np.array([0.0, 0.0]) 
        self.most_similar_position = None

        self.map_size = 10000  # Adjust the size of the map as needed
        self.map = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
        self.map_scale = 1  # Adjust the scale of the map grid
        self.map_display_name = "Robot Map"
        cv2.namedWindow(self.map_display_name)
        

        super(KeyboardPlayerPyGame, self).__init__()

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.position = np.array([0.0, 0.0]) 
        self.distance_count = 0  # To track the distance moved
        self.move_distance = 0

        self.current_angle = 0  # Add this line to store the current angle

        self.find = False

        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }
    

    def store_captured_image(self, image, camera_position):
        # Store the entire captured image
        # print("self.captured_images: ", self.captured_images)
        self.captured_images.append((camera_position, image))

    def compare_with_target_features(self):
        # Load the target image
        target_image = cv2.imread(self.target_image_path, cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create()

        # Initialize variables to track the best match
        most_similar_image = None
        highest_similarity = 0
        most_similar_position = None

        # Extract features from the target image
        target_keypoints, target_descriptors = orb.detectAndCompute(target_image, None)

        for i, (camera_position, captured_image) in enumerate(self.captured_images):
            # Extract features from the captured image
            keypoints, descriptors = orb.detectAndCompute(captured_image, None)

            # Create a BFMatcher (Brute Force Matcher) with Hamming distance
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Match the descriptors
            matches = bf.match(target_descriptors, descriptors)

            matches = sorted(matches, key=lambda x: x.distance)

            # Calculate similarity based on the number of matches
            similarity = 1 - (matches[0].distance / len(target_descriptors))

            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_image = captured_image  # Store the most similar image
                most_similar_position = camera_position
                most_similar_index = i

        if most_similar_image is not None:
            print(f"The most similar image is found with a similarity of {highest_similarity}")
            print(f"Index of matched image: {most_similar_index}")
            print(f"Camera Position: {most_similar_position}")
            self.most_similar_position = most_similar_position
            self.find = True
            cv2.imshow("Target Image", target_image)
            cv2.imshow("Matched Captured Image", most_similar_image)  # Display the most similar image
            cv2.waitKey(0)

            
        else:
            print("No similar image found in the dataset.")
        
    def draw_guidance_on_image(self):
        # Check if we have a valid similar position to guide towards
        if self.most_similar_position is None:
            return "No guidance available."

        # Calculate the direction vector from current to most similar position
        direction = self.most_similar_position - self.position
        guidance_text_x = ""
        guidance_text_y = ""

        # Check the x-coordinate difference
        if direction[0] > 0:
            guidance_text_x += f"Increase X (right: {abs(direction[0]):.2f}). "
        elif direction[0] < 0:
            guidance_text_x += f"Decrease X (left: {abs(direction[0]):.2f}). "

        # Check the y-coordinate difference
        if direction[1] > 0:
            guidance_text_y += f"Increase Y (forward: {abs(direction[1]):.2f})."
        elif direction[1] < 0:
            guidance_text_y += f"Decrease Y (backward: {abs(direction[1]):.2f})."

        guidance_text = guidance_text_x + guidance_text_y
        # If there's no significant difference in either direction, indicate that the current position is close enough
        if guidance_text == "":
            guidance_text = "Current position is close to the target!"

        return guidance_text, guidance_text_x, guidance_text_y

    def pre_exploration(self):
        self.store_captured_images_flag = True

    def pre_navigation(self):
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
                else:
                    self.show_target_images()
                    self.compare_with_target_features()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]

        if self.last_act & Action.FORWARD or self.last_act & Action.BACKWARD:
            self.calculate_move_distance()
            self.get_position()
        
        if self.last_act & Action.LEFT or self.last_act & Action.RIGHT:
            self.calculate_rotation_angle()

        return self.last_act
    
    # def get_position(self):
    #     angle_in_radians = math.radians(self.current_angle)  # Convert angle from degrees to radians.
    #     direction = 1 if self.last_act & Action.FORWARD else -1 if self.last_act & Action.BACKWARD else 0
    #     move_vector = direction * self.move_distance * np.array([math.cos(angle_in_radians), math.sin(angle_in_radians)])
    #     self.position += move_vector  # Update the position.
    #     # print(f"Current position after move: (X: {self.position[0]:.2f}, Y: {self.position[1]:.2f})")

    def world_to_map_indices(self, world_position):
        # Convert world coordinates to map indices
        map_x = int(world_position[0] * self.map_scale + self.map_size / 2)
        map_y = int(world_position[1] * self.map_scale + self.map_size / 2)
        return map_x, map_y

    def get_position(self):
        angle_in_radians = math.radians(self.current_angle)
        direction = 1 if self.last_act & Action.FORWARD else -1 if self.last_act & Action.BACKWARD else 0
        move_vector = direction * self.move_distance * np.array([math.cos(angle_in_radians), math.sin(angle_in_radians)])
        new_position = self.position + move_vector

        # Update the map based on the movement
        self.update_map(self.position, new_position)

        self.position = new_position
        # print(f"Current position after move: (X: {self.position[0]:.2f}, Y: {self.position[1]:.2f})")


    def calculate_rotation_angle(self):
        if self.last_act & Action.LEFT:
            self.current_angle -= self.turning_angle
            if -370 <= self.current_angle <= -350:
                self.current_angle = 0  # Reset to 0 if within the specified range
        
        if self.last_act & Action.RIGHT:
            self.current_angle += self.turning_angle

            # Check if the current angle is between 350 and 370 degrees
            if 350 <= self.current_angle <= 370:
                self.current_angle = 0  # Reset to 0 if within the specified range

        self.current_angle = int(self.current_angle)
        # Print the current angle to the console - or you can display it on the screen as needed
        # print(f"Current angle: {self.current_angle}")


    
    def calculate_move_distance(self):
        # Assuming a simple step count as distance
        if self.last_act & Action.FORWARD:
            self.distance_count += 1
        elif self.last_act & Action.BACKWARD:
            self.distance_count -= 1  # Assuming negative for backward movement
        else:
            self.distance_count = 0  # Set to 0 if neither forward nor backward is pressed


        # To calculate the distance moved, you'd multiply the distance_count by your predefined step size
        self.move_distance = self.distance_count * self.move_step
        # print(f"Distance moved: {self.move_distance} units")

    def show_target_images(self):
        # reset
        self.position = np.array([0.0, 0.0]) 
        self.distance_count = 0  
        self.move_distance = 0
        self.current_angle = 0  


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

        selected_target = targets[0]

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

        self.target_image_path = 'target_temp.png'
        cv2.imwrite(self.target_image_path, selected_target)

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))
        
        if self.store_captured_images_flag: 
            self.store_captured_image(self.fpv, self.position.astype(int))

        self.display_map()
        
        cv2.putText(self.fpv, str(self.position.astype(int)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        

            
        if self.find is True:
            guid, guid_x, guid_y = self.draw_guidance_on_image()
            # Adjust font size here by changing the `1` to a smaller value like `0.5`
            cv2.putText(self.fpv, guid_x, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(self.fpv, guid_y, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            


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
        pygame.display.update()

    def detect_walls(self):
        # Your wall detection logic goes here
        # Example: Use a simple thresholding approach for demonstration purposes
        _, thresholded = cv2.threshold(self.fpv, 200, 255, cv2.THRESH_BINARY)

        # Check if walls are detected
        walls_detected = np.any(thresholded > 0)

        # Update the map based on the detection result
        if walls_detected:
            self.update_map_with_walls(thresholded)
        else:
            self.display_no_walls()

    def update_map(self, old_position, new_position):
        # Convert coordinates to map indices
        old_x, old_y = self.world_to_map_indices(old_position)
        new_x, new_y = self.world_to_map_indices(new_position)

        # Ensure the indices are within the map boundaries
        old_x = max(0, min(old_x, self.map_size - 1))
        old_y = max(0, min(old_y, self.map_size - 1))
        new_x = max(0, min(new_x, self.map_size - 1))
        new_y = max(0, min(new_y, self.map_size - 1))

        # Draw a continuous curve between old and new positions
        cv2.line(self.map, (old_x, old_y), (new_x, new_y), 255, 2)

        # Draw a small circle at the new position to ensure continuity
        cv2.circle(self.map, (new_x, new_y), 1, 255, -1)

    def display_map(self):
        # Resize the map for better visualization
        map_display = cv2.resize(self.map, (500, 500), interpolation=cv2.INTER_NEAREST)
        map_display = cv2.cvtColor(map_display, cv2.COLOR_GRAY2BGR)

        # Draw a circle representing the robot's current position on the map
        map_center = (int(self.map_size * self.map_scale / 2), int(self.map_size * self.map_scale / 2))
        robot_center = (int(self.position[0] * self.map_scale + map_center[0]),
                        int(self.position[1] * self.map_scale + map_center[1]))
        cv2.circle(map_display, robot_center, 5, (0, 0, 255), -1)

        # Resize the fpv array to match the height of the map_display array
        fpv_resized = cv2.resize(self.fpv, (self.fpv.shape[1], map_display.shape[0]))

        # Display the map alongside the first-person view
        combined_image = map_display
        cv2.imshow(self.map_display_name, combined_image)
        cv2.waitKey(1)


if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())