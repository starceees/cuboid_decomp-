import numpy as np
import pygame
from pygame.locals import QUIT

# Initialize Pygame
pygame.init()

# Set up display
screen_size = (500, 500)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('3D Map Visualization')

# Initialize variables
x, y, theta = 0, 0, 0  # Initial pose
map_points = np.zeros((0, 2))  # Initialize empty map points

# Function to update pose based on keyboard input
def update_pose():
    global x, y, theta

    step_size = 1.0  # Adjust as needed
    angle_step = np.radians(5)  # Adjust as needed

    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        x += step_size * np.cos(theta)
        y += step_size * np.sin(theta)
    elif keys[pygame.K_DOWN]:
        x -= step_size * np.cos(theta)
        y -= step_size * np.sin(theta)
    elif keys[pygame.K_LEFT]:
        theta += angle_step
    elif keys[pygame.K_RIGHT]:
        theta -= angle_step

# Function to visualize the map
def visualize_map():
    global map_points

    screen.fill((255, 255, 255))  # Clear the screen

    # Convert map points to screen coordinates
    map_points_screen = ((map_points + 5) * 50).astype(int)

    for point in map_points_screen:
        pygame.draw.circle(screen, (0, 0, 0), tuple(point), 2)

    pygame.display.flip()

# Main loop to capture keyboard input and update the pose
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    # Update pose and map based on key
    update_pose()
    map_points = np.vstack([map_points, [x, y]])

    # Visualize the map
    visualize_map()

    clock.tick(30)  # Adjust as needed for the desired frame rate

pygame.quit()
