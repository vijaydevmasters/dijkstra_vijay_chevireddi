from queue import PriorityQueue


def move_robot(position, action):
    return (position[0] + action[0], position[1] + action[1])

def get_neighbors(position):
    actions = {
        (1, 0): 1,   # Right
        (-1, 0): 1,  # Left
        (0, -1): 1,  # Down
        (0, 1): 1,   # Up
        (1, 1): 1.4, # Up-Right
        (-1, 1): 1.4,# Up-Left
        (1, -1): 1.4,# Down-Right
        (-1, -1): 1.4# Down-Lefts
        
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
    open_set = PriorityQueue()
    open_set.put((0, start_point))
    came_from = {}
    cost_to_come = {start_point: 0}
    
    while not open_set.empty():
        current_cost, current_node = open_set.get()


        if current_cost > cost_to_come.get(current_node, float('inf')):
            continue

        if current_node == goal_point:
            return backtrack(came_from, current_node), cost_to_come[current_node]

        for neighbor, move_cost in get_neighbors(current_node):
            new_cost = cost_to_come[current_node] + move_cost
            if neighbor not in cost_to_come or new_cost < cost_to_come[neighbor]:
                cost_to_come[neighbor] = new_cost
                priority = new_cost
                open_set.put((priority, neighbor))
                came_from[neighbor] = current_node

    return [], float('inf')  # Goal not found


# Get the start and goal points from the user
x_s = int(input("Enter x-coordinate of start point of mobile robot: "))
y_s = int(input("Enter y-coordinate of start point of mobile robot: "))
x_g = int(input("Enter x-coordinate of goal point of mobile robot: "))
y_g = int(input("Enter y-coordinate of goal point of mobile robot: "))

start_point = (x_s, y_s)
goal_point = (x_g, y_g)


path = dijkstra(start_point, goal_point)
print(path)