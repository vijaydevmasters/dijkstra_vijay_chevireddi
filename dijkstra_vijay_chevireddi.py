from queue import PriorityQueue
import cv2
import numpy as np
import math

class MapDrawer:
    def __init__(self, width, height, cell_size=5):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.image = np.ones((height, width, 3), dtype=np.uint8) * 255
        self.obstacle_color = (0, 0, 0)
        self.start_color = (0, 0, 255)
        self.goal_color = (0, 255, 0)
        self.explored_color = (255, 0, 0)
        self.path_color = (255, 255, 0)
        self.inflated_color = (0, 0, 200)

    def convert_to_image_coordinates(self, grid_x, grid_y):
        img_x = grid_x
        img_y = self.height - grid_y
        return img_x, img_y

    def render_rectangle(self, lower_left_corner, upper_right_corner, color, fill=True):
        fill_value = -1 if fill else 1
        cv2.rectangle(self.image, lower_left_corner, upper_right_corner, color, fill_value)

    def render_hexagon(self, center_point, side_length, color, border_thickness=1):
        hexagon_vertices = []
        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = math.pi / 180 * angle_deg
            vertex_x = center_point[0] + side_length * math.cos(angle_rad)
            vertex_y = center_point[1] + side_length * math.sin(angle_rad)
            hexagon_vertices.append((int(vertex_x), int(vertex_y)))
        cv2.fillPoly(self.image, [np.array(hexagon_vertices)], (0,0,0))
        
        if border_thickness > 0:
            cv2.polylines(self.image, [np.array(hexagon_vertices)], isClosed=True, color=color, thickness=border_thickness)

    def add_obstacles_to_map(self, obstacle_list):
        for obstacle in obstacle_list:
            obstacle_shape = obstacle['type']
            obstacle_color = obstacle.get('fill_color', self.obstacle_color)
            is_filled = obstacle.get('fill', -1) == -1
            if obstacle_shape == 'r':
                bottom_left = self.convert_to_image_coordinates(*obstacle['lower_left'])
                top_right = self.convert_to_image_coordinates(*obstacle['upper_right'])
                self.render_rectangle(bottom_left, top_right, obstacle_color, is_filled)
            elif obstacle_shape == 'h':
                center = self.convert_to_image_coordinates(*obstacle['hex_center'])
                side_length = obstacle['hex_side']
                
                border_thickness = 4 if not is_filled else 0
                self.render_hexagon(center, side_length, obstacle_color, border_thickness)


    def draw_cell(self, coord, color):
        x, y = coord
        y = self.height - y * self.cell_size  # Adjust for cell size and invert y-axis
        x = x * self.cell_size  # Adjust x-coordinate for cell size
        cv2.circle(self.image, (x, y), self.cell_size // 2, color, -1)

    def visualize_path(self, path):
        for coord in path:
            self.draw_cell(coord, self.path_color)
            cv2.imshow("Dijkstra", self.image)
            cv2.waitKey(50)

# Initialize the MapDrawer
grid_width = 1200
grid_height = 500
map_drawer = MapDrawer(grid_width, grid_height)

# "r = rectangle", "h = hexagon"
obstacles_with_bloat = [{'type': 'r', 'lower_left': (95, 95), 'upper_right': (180, 500), 'fill_color': map_drawer.inflated_color, 'fill': -1},
    {'type': 'r', 'lower_left': (100, 100), 'upper_right': (175, 500), 'fill_color': (0, 0, 0), 'fill': -1},
    {'type': 'r', 'lower_left': (270, 0), 'upper_right': (355, 405), 'fill_color': map_drawer.inflated_color, 'fill': -1},
    {'type': 'r', 'lower_left': (275, 0), 'upper_right': (350, 400), 'fill_color': (0, 0, 0), 'fill': -1},
    {'type': 'r', 'lower_left': (1020, 45), 'upper_right': (1105, 455), 'fill_color': map_drawer.inflated_color, 'fill': -1},
    {'type': 'r', 'lower_left': (895, 45), 'upper_right': (1105, 130), 'fill_color': map_drawer.inflated_color, 'fill': -1},
    {'type': 'r', 'lower_left': (900, 50), 'upper_right': (1100, 125), 'fill_color': (0, 0, 0), 'fill': -1},
    {'type': 'r', 'lower_left': (895, 370), 'upper_right': (1105, 455), 'fill_color': map_drawer.inflated_color, 'fill': -1},
    {'type': 'r', 'lower_left': (900, 375), 'upper_right': (1100, 450), 'fill_color': (0, 0, 0), 'fill': -1},
    {'type': 'r', 'lower_left': (1025, 50), 'upper_right': (1100, 450), 'fill_color': (0, 0, 0), 'fill': -1},
    {'type': 'r', 'lower_left': (0, 0), 'upper_right': (1200, 5), 'fill_color': map_drawer.inflated_color, 'fill': -1},
    {'type': 'r', 'lower_left': (0, 0), 'upper_right': (5, 500), 'fill_color': map_drawer.inflated_color, 'fill': -1},
    {'type': 'r', 'lower_left': (1195, 0), 'upper_right': (1200, 500), 'fill_color': map_drawer.inflated_color, 'fill': -1},
    {'type': 'r', 'lower_left': (0, 495), 'upper_right': (1200, 500), 'fill_color': map_drawer.inflated_color, 'fill': -1},
    {'type': 'h', 'hex_center': (650, 250), 'hex_side': 150, 'fill_color': map_drawer.inflated_color, 'fill': 4}]

map_drawer.add_obstacles_to_map(obstacles_with_bloat)

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


def backtrack(parent, current):
    path = [current]
    while current in parent:
        current = parent[current]
        path.append(current)
    path.reverse()
    return path


def dijkstra(start_point, goal_point):
    map_drawer.draw_cell(start_point, map_drawer.start_color)
    map_drawer.draw_cell(goal_point, map_drawer.goal_color)

    open_list = PriorityQueue()
    open_list.put((0, start_point))
    parent = {}
    closed_list = {start_point: 0}

    while not open_list.empty():
        cv2.imshow('Dijkstra', map_drawer.image)
        if cv2.waitKey(1) == ord('q'):
            break
        current_cost, current_node = open_list.get()

        # As we cannot mutate elements of a tuple inside priority queue. Just skipping the non-updated elemets of the nodes in Open List.
        if current_cost > closed_list.get(current_node, float("inf")):
            continue

        if current_node == goal_point:
            path = backtrack(parent, current_node)
            map_drawer.visualize_path(path)
            return backtrack(parent, current_node), closed_list[current_node]

        for neighbor, move_cost in get_neighbors(current_node):
            new_cost = closed_list[current_node] + move_cost
            if neighbor not in closed_list or new_cost < closed_list[neighbor]:
                closed_list[neighbor] = new_cost
                open_list.put((new_cost, neighbor))
                parent[neighbor] = current_node
                map_drawer.draw_cell(neighbor, map_drawer.explored_color)

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
    cv2.imshow('Dijkstra', map_drawer.image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()