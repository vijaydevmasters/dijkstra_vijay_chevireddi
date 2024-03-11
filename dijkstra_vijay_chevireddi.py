from queue import PriorityQueue
import cv2
import numpy as np
import math

class MapDrawer:
    """
    Handles drawing of the map, obstacles, and paths for pathfinding visualization.
    """
    def __init__(self, width, height, cell_size=5):
        # Initialize map properties
        self.width = width
        self.height = height
        self.cell_size = cell_size
        # Create a white image for the map background
        self.image = np.ones((height, width, 3), dtype=np.uint8) * 255
        # Define colors for different elements on the map
        self.obstacle_color = (0, 0, 0)
        self.start_color = (0, 0, 255)
        self.goal_color = (0, 255, 0)
        self.explored_color = (255, 0, 0)
        self.path_color = (255, 255, 0)
        self.inflated_color = (0, 0, 200)
        
    def convert_to_image_coordinates(self, grid_x, grid_y):
        """
        Convert grid coordinates to image coordinates (flipping the y-axis).
        """
        img_x = grid_x
        img_y = self.height - grid_y
        return img_x, img_y

    def render_rectangle(self, lower_left_corner, upper_right_corner, color, fill=True):
        """
        Draw a rectangle on the map.
        """
        fill_value = -1 if fill else 1
        cv2.rectangle(self.image, lower_left_corner, upper_right_corner, color, fill_value)

    def render_hexagon(self, center_point, side_length, color, border_thickness=1):
        """
        Draw a hexagon on the map.
        """
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
        """
        Interpret and draw obstacles on the map.
        """
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
        """
        Draw a cell on the map.
        """
        grid_x, grid_y = coord
        # print(grid_x, grid_y)
        # Draw the circle for the cell using the correct image coordinates
        cv2.circle(self.image, (grid_x, grid_y),1,  color, -1)



    def visualize_path(self, path):
        for coord in path:
            self.draw_cell(coord, self.path_color)
            cv2.imshow("Dijkstra", self.image)
            cv2.waitKey(50)

    def is_valid_neighbor(self, point):
        """
        Check if a point is a valid, unobstructed map location.
        """
        # print(point)
        x, y = point
        if 0 <= x < self.width and 0 <= y < self.height and self.image[y - 1, x, 0] == 255:
            return True
        return False


    
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
    actions = {(1, 0): 1,(-1, 0): 1,(0, -1): 1,(0, 1): 1,(1, 1): 1.4,(-1, 1): 1.4,(1, -1): 1.4,(-1, -1): 1.4,
    }
    neighbors = []
    for action, cost in actions.items():
        new_position = move_robot(position, action)
        if map_drawer.is_valid_neighbor(new_position):
            neighbors.append((new_position, cost))
    return neighbors


def backtrack(parent, current):
    path = [current]
    while current in parent:
        current = parent[current]
        path.append(current)
    path.reverse()# Reverse the path to start->goal order
    return path


def get_coordinate_input(prompt_message):
    while True:
        user_input = input(f"{prompt_message}: ")
        
        try:
            input_x, input_y = map(int, user_input.split(','))
            converted_x, converted_y = map_drawer.convert_to_image_coordinates(input_x, input_y)
        except ValueError:
            print("Invalid input format. Please enter coordinates in the format x,y.")
            continue

        image_y = map_drawer.height - converted_y
        if 0 <= converted_x < map_drawer.width and 0 <= image_y < map_drawer.height and map_drawer.image[image_y, converted_x, 0] == 255:
            return converted_x, converted_y
        else:
            print("Point is invalid or lies on an obstacle. Please try again.")


def visualise_planning_backtracking(explored_nodes, path, goal):

    explored_count = 0
    cv2.circle(map_drawer.image, (goal_point[0], goal_point[1]), 5, (0, 0, 255), -1) # Goal Node
    cv2.circle(map_drawer.image, (start_point[0], start_point[1]), 5, (255, 0, 0), -1) # Start Node
    
    for node in explored_nodes:
        map_drawer.image[node[1], node[0]] = (168, 185, 203)
        if explored_count % 160 == 0:
            cv2.imshow("Planning with shortest path", map_drawer.image) # Drawing explored Nodes
            out.write(map_drawer.image)
            cv2.waitKey(1)
        explored_count += 1

    for point in path:
        cv2.circle(map_drawer.image, (point[0], point[1]), 2, (108, 79, 11), -1)
        cv2.imshow("Planning with shortest path", map_drawer.image)
        out.write(map_drawer.image)
        cv2.waitKey(1)

def dijkstra(start_point, goal_point):
    # Priority queue for the open set with initial node having cost 0
    open_set = PriorityQueue()
    open_set.put((0, start_point))
    came_from = {}# To reconstruct the path
    cost_to_come = {start_point: 0} # Cost from start to the node
    explored = []
    while not open_set.empty():
        current_cost, current_node = open_set.get()

          # As we cannot mutate elements of a tuple inside priority queue. Just skipping the unupdated elemets of the nodes in Open List.
        if current_cost > cost_to_come.get(current_node, float('inf')):
            continue

        if current_node == goal_point:
            # If goal is reached, visualize the result and return the path and its cost
            visualise_planning_backtracking(explored, backtrack(came_from, current_node), current_node)
            return backtrack(came_from, current_node), cost_to_come[current_node]

        for neighbor, move_cost in get_neighbors(current_node):
            new_cost = cost_to_come[current_node] + move_cost
             # If new cost is lower, update the cost and path for this neighbor
            if neighbor not in cost_to_come or new_cost < cost_to_come[neighbor]:
                cost_to_come[neighbor] = new_cost
                priority = new_cost
                open_set.put((priority, neighbor))
                came_from[neighbor] = current_node
                explored.append(neighbor)

# Start Point
start_point = get_coordinate_input("Enter start point (x, y) in this way : 11,11 ")
# Goal Point
goal_point = get_coordinate_input("Enter goal point (x, y) in this way : 1190,400 ")

# Output Video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('creddy_video.mp4', fourcc, 40.0, (map_drawer.width, map_drawer.height))

path = dijkstra(start_point, goal_point)
print(path)

while True:
    cv2.imshow('Dijkstra', map_drawer.image)
    out.write(map_drawer.image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
cv2.destroyAllWindows()