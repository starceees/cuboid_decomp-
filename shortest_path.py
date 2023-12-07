import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Example binary map (0s and 1s representing obstacles and free spaces)
binary_map = np.array([
    [0, 0, 1, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 1, 0],
])

# Function to check if a position is within bounds and not an obstacle
def is_valid_position(pos):
    x, y = pos
    return 0 <= x < binary_map.shape[0] and 0 <= y < binary_map.shape[1] and binary_map[x][y] == 0

# Function to get neighboring positions
def get_neighbors(pos):
    x, y = pos
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]  # Adjacent positions
    valid_neighbors = [neighbor for neighbor in neighbors if is_valid_position(neighbor)]
    return valid_neighbors

# A* algorithm implementation
def astar(start, goal):
    graph = nx.Graph()
    graph.add_nodes_from([(i, j) for i in range(binary_map.shape[0]) for j in range(binary_map.shape[1]) if binary_map[i][j] == 0])

    for i in range(binary_map.shape[0]):
        for j in range(binary_map.shape[1]):
            if binary_map[i][j] == 0:
                neighbors = get_neighbors((i, j))
                for neighbor in neighbors:
                    graph.add_edge((i, j), neighbor, weight=1)  # Assuming uniform cost for movement

    path = nx.astar_path(graph, start, goal)
    return path

# Example start and goal positions (change as needed)
start_position = (0, 0)
goal_position = (4, 4)

# Finding the shortest path
shortest_path = astar(start_position, goal_position)
print("Shortest Path:", shortest_path)

# Invert the colors of the binary map (0 to 1 and 1 to 0)
inverted_binary_map = np.where(binary_map == 0, 1, 0)

# Visualizing the inverted binary map with the shortest path
plt.figure(figsize=(8, 8))
plt.imshow(inverted_binary_map, cmap='gray')

# Plotting the shortest path on the map
path_x, path_y = zip(*shortest_path)
plt.plot(path_y, path_x, color='red', marker='o')

# Marking start and goal positions
plt.plot(start_position[1], start_position[0], 'go')  # Green for start
plt.plot(goal_position[1], goal_position[0], 'bo')  # Blue for goal

plt.title('Inverted Binary Map with Shortest Path')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()
