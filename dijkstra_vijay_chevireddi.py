from queue import PriorityQueue
import cv2
import numpy as np

GRID_SIZE = 100
CELL_SIZE = 5
START_COLOR = (0, 0, 255)
GOAL_COLOR = (0, 255, 0)
EXPLORED_COLOR = (255, 0, 0)
PATH_COLOR = (255, 255, 0)

image = np.ones((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE, 3), dtype=np.uint8) * 255
cv2.namedWindow("Dijkstra", cv2.WINDOW_AUTOSIZE)

# Dijkstra Algorithm OpenCV fuunctions


def visualize_path(path):
    for point in path:

        x, y = point
        y = GRID_SIZE - 1 - y
        top_left = (x * CELL_SIZE, y * CELL_SIZE)
        bottom_right = ((x + 1) * CELL_SIZE - 1, (y + 1) * CELL_SIZE - 1)
        cv2.rectangle(image, top_left, bottom_right, (0, 200, 0), -1)
        cv2.imshow("Dijkstra", image)
        cv2.waitKey(50)


# Dijkstra Algorithm Helper Functions
def move_robot(position, action):
    return (position[0] + action[0], position[1] + action[1])


def get_neighbors(position):
    actions = {
        (1, 0): 1,  # Right
        (-1, 0): 1,  # Left
        (0, -1): 1,  # Down
        (0, 1): 1,  # Up
        (1, 1): 1.4,  # Up-Right
        (-1, 1): 1.4,  # Up-Left
        (1, -1): 1.4,  # Down-Right
        (-1, -1): 1.4,  # Down-Left
    }
    neighbors = []
    for action, cost in actions.items():
        new_position = move_robot(position, action)
        neighbors.append((new_position, cost))
    return neighbors


def backtrack(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def dijkstra(start_point, goal_point):
    open_list = PriorityQueue()
    open_list.put((0, start_point))
    came_from = {}
    closed_list = {start_point: 0}

    while not open_list.empty():
        cv2.imshow('Dijkstra', image)
        if cv2.waitKey(1) == ord('q'):
            break
        current_cost, current_node = open_list.get()

        # As we cannot mutate elements of a tuple inside priority queue. Just skipping the unupdated elemets of the nodes in Open List.
        if current_cost > closed_list.get(current_node, float("inf")):
            continue

        if current_node == goal_point:
            path = backtrack(came_from, current_node)
            visualize_path(path)
            return backtrack(came_from, current_node), closed_list[current_node]

        for neighbor, move_cost in get_neighbors(current_node):
            new_cost = closed_list[current_node] + move_cost
            if neighbor not in closed_list or new_cost < closed_list[neighbor]:
                closed_list[neighbor] = new_cost
                priority = new_cost
                open_list.put((priority, neighbor))
                came_from[neighbor] = current_node

    return [], float("inf")  # Goal not found


# Get the start and goal points from the user
x_s = int(input("Enter x-coordinate of start point of mobile robot: "))
y_s = int(input("Enter y-coordinate of start point of mobile robot: "))
x_g = int(input("Enter x-coordinate of goal point of mobile robot: "))
y_g = int(input("Enter y-coordinate of goal point of mobile robot: "))

start_point = (x_s, y_s)
goal_point = (x_g, y_g)


path = dijkstra(start_point, goal_point)
print(path)

while True:
    cv2.imshow('Dijkstra', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()