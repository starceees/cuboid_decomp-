from vis_nav_game import Player, Action
import pygame
import cv2
import numpy as np
import matplotlib.pyplot as plt

class KeyboardPlayerPyGame(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.robot_pose = [0, 0, 90] 
        self.robot_poses = []  # Store robot poses
        self.directions = []
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
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]

        self.update_robot_pose()
                    
        return self.last_act
    
    def update_robot_pose(self):
        # Update robot pose based on actions
        if self.last_act == Action.LEFT:
            self.robot_pose[2] += 90  # Rotate 90 degrees counterclockwise
            self.directions.append(2)
        elif self.last_act == Action.RIGHT:
            self.robot_pose[2] -= 90  # Rotate 90 degrees clockwise
            self.directions.append(4)
        elif self.last_act == Action.FORWARD:
            self.robot_pose[0] += 0.1 * np.cos(np.deg2rad(self.robot_pose[2]))  # Move forward in x direction
            self.robot_pose[1] += 0.1 * np.sin(np.deg2rad(self.robot_pose[2]))  # Move forward in y direction
            self.directions.append(1)
        elif self.last_act == Action.BACKWARD:
            self.robot_pose[0] -= 0.1 * np.cos(np.deg2rad(self.robot_pose[2]))  # Move backward in x direction
            self.robot_pose[1] -= 0.1 * np.sin(np.deg2rad(self.robot_pose[2]))  # Move backward in y direction
            self.directions.append(5)

        # Store robot pose
        self.robot_poses.append(self.robot_pose.copy())

    def display_path(self):
        # Plot and display the robot path
        x = [pose[0] for pose in self.robot_poses]
        y = [pose[1] for pose in self.robot_poses]

        plt.plot(x, y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Robot Path')

        # Display pose values
        for i, pose in enumerate(self.robot_poses):
            print(pose[0], pose[1])

        for dirs in enumerate(self.directions):
            print(dirs[1])

        start_pose = self.robot_poses[0]
        end_pose = self.robot_poses[-1]
        plt.plot(start_pose[0], start_pose[1], 'go', label='Start')
        plt.plot(end_pose[0], end_pose[1], 'ro', label='End')
        plt.legend()

        plt.show()

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

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

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

        def convert_opencv_img_to_pygame(opencv_image):
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()

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

    def move_autonomously(self):
        for dir in self.directions:
            if dir[1] == 1:
                return Action.FORWARD
            elif dir[1] == 2:
                return Action.LEFT
            elif dir[1] == 4:
                return Action.RIGHT
            else:
                return Action.IDLE

if __name__ == "__main__":
    import vis_nav_game

    player = KeyboardPlayerPyGame()
    vis_nav_game.play(the_player=player)

    # Display the path after the game is finished
    player.display_path()

    # Move the robot autonomously on the saved path
    player.move_autonomously()