import getopt
import sys
import numpy
from os import path
import random
import time
import torch
from torch import nn
from torch.nn import functional
from contextlib import redirect_stderr

NUM_DEBUG_ITR = 2
debug_itr = 1

WIDTH, HEIGHT = 24, 13

ROBOT_SAMPLE_SIZE = 4
EPSILON = 0.5
LEARNING_RATE_HQ = 0.1
LEARNING_RATE_ROBOT = 0.001 # LEARNING_RATE_HQ / ROBOT_SAMPLE_SIZE

robot_gamma = 0.5
HQ_gamma = 0.9
r = 1.0
device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
file_suffix = "_v0"
debug = False
learning = False



# This code will be run upon startup
def input_help():
    print(file_suffix, file=sys.stderr, flush=True)
    print("Usage: ai.py -f <file_suffix> -r <learning_rate_weight> [-l] [-d]", file=sys.stderr, flush=True)
    sys.exit(2)

try:
    opts, args = getopt.getopt(sys.argv[1:], "hlr:f:d")
except getopt.GetoptError:
    input_help()

for opt, arg in opts:
    if opt == "-f":
        file_suffix = arg
    if opt == "-h":
        input_help()
    if opt == "-d":
        debug = True
    if opt == "-l":
        learning = True
    if opt == "-r":
        r = float(arg)

with open("ai" + file_suffix + ".log", "w") as stderr, redirect_stderr(stderr):

    # NOTE game_board_inputs DOES NOT have gradiants for a REASON!!!
    HQ_game_board_input = torch.zeros(4, HEIGHT, WIDTH, device=device)
    robot_game_board_input = torch.zeros(4, HEIGHT, WIDTH, device=device)
    my_matter_input = torch.zeros(1, device=device, requires_grad=learning)
    opp_matter_input = torch.zeros(1, device=device, requires_grad=learning)
    true_width, true_height = [int(i) for i in input().split()] if not debug else [21, 10]
    HQ_prev_state = torch.clone(HQ_game_board_input)
    robot_prev_state = torch.clone(robot_game_board_input)
    prev_action = 0
    prev_my_matter = torch.clone(my_matter_input)
    prev_opp_matter = torch.clone(opp_matter_input)
    prev_coordinates = []
    prev_moves = []
    robot_rewards_sheet = []
    prev_destination_owners = []
    num_my_tiles, num_opp_tiles, prev_num_my_tiles, prev_num_opp_tiles = 0, 0, 0, 0

    raw_game_board_input = numpy.empty((HEIGHT, WIDTH, 7), dtype=numpy.float32)
    scaled_game_board_input = numpy.zeros((4, HEIGHT, WIDTH), dtype=numpy.float32)
    for i in range(HEIGHT):
        for j in range(WIDTH):
            scaled_game_board_input[1][i][j] = -1 # Need to set the owner to no-one
    my_robots_coords = []
    ideal_build_coords = []
    available_build_coords = []
    ideal_spawn_coords = []
    available_spawn_coords = []
    curr_score = 0
    start_left = True

    is_first_loop = True
    n = 1

    class HQNeuralNetwork(nn.Module):
        def __init__(self):
            super(HQNeuralNetwork, self).__init__()
            self.convolution1 = nn.Conv2d(4, 16, 3, groups=4, device=device) # Was padding=1
            self.pool = nn.MaxPool2d(2)
            self.relu = nn.ReLU()
            self.flatten = nn.Flatten(start_dim=0)
            self.linear1 = nn.Linear(self.num_inputs_from_conv() + 2, 20, device=device)
            self.linear2 = nn.Linear(20, 3, device=device)
            self.softmax = nn.Softmax(dim=0)

        def num_inputs_from_conv(self):
            board = torch.rand((4, HEIGHT, WIDTH), device=device)
            board = self.convolution1(board)
            board = self.relu(board)
            board = self.pool(board)
            return self.flatten(board).size(dim=0)
        
        def forward(self, game_board, my_matter, opp_matter):
            board = self.convolution1(game_board)
            board = self.relu(board)
            board = self.pool(board)
            all_inputs = torch.cat([self.flatten(board), my_matter, opp_matter])
            all_inputs = self.linear1(all_inputs)
            all_inputs = self.relu(all_inputs)
            all_inputs = self.linear2(all_inputs)
            all_inputs = self.relu(all_inputs)
            all_inputs = self.softmax(all_inputs)
            return all_inputs
        
        def learn(self, reward, action, curr_q_value, next_q_value):
            target = curr_q_value.clone()
            target.data[action] = reward + HQ_gamma * torch.max(next_q_value)
            target.detach()
            
            #loss = functional.smooth_l1_loss(curr_q_value[action], torch.tanh(torch.tensor(reward)).item() + HQ_gamma * torch.max(next_q_value))
            board_brain.zero_grad()
            loss = functional.smooth_l1_loss(curr_q_value, target)
            #board_brain.zero_grad()
            
            loss.backward()
            for param in board_brain.parameters(): param.data.sub_(param.grad.data * LEARNING_RATE_HQ * r)

    class RobotNeuralNetwork(nn.Module):
        def __init__(self):
            super(RobotNeuralNetwork, self).__init__()
            self.convolution1 = nn.Conv2d(4, 16, 3, groups=4, device=device)
            self.pool = nn.MaxPool2d(2)
            self.relu = nn.ReLU()
            self.flatten = nn.Flatten(start_dim=0)
            self.linear1 = nn.Linear(self.num_inputs_from_conv() + 2, 20, device=device)
            self.linear2 = nn.Linear(20, 20, device=device)
            self.linear3 = nn.Linear(20, 5, device=device)
            self.softmax = nn.Softmax(dim=0)
        
        def num_inputs_from_conv(self):
            board = torch.rand((4, HEIGHT, WIDTH), device=device)
            board = self.convolution1(board)
            board = self.relu(board)
            board = self.pool(board)
            return self.flatten(board).size(dim=0)
        
        def forward(self, game_board, position):
            board = self.convolution1(game_board)
            board = self.relu(board)
            board = self.pool(board)
            all_inputs = torch.cat([self.flatten(board), position])
            all_inputs = self.linear1(all_inputs)
            all_inputs = self.relu(all_inputs)
            all_inputs = self.linear2(all_inputs)
            all_inputs = self.relu(all_inputs)
            all_inputs = self.linear3(all_inputs)
            all_inputs = self.softmax(all_inputs)
            return all_inputs
        
        def learn(self, prev_game_board, prev_coordinates, prev_moves, rewards, next_game_board):
            if len(prev_coordinates) > 0:
                total_prev_q_value = 0.0
                total_target = 0.0
                for j in range(min(len(prev_coordinates), ROBOT_SAMPLE_SIZE)):
                    i = torch.randint(0, len(prev_coordinates), (1,))
                    prev_q_value = robot_brain(prev_game_board, prev_coordinates[i])
                    coord_y = round(prev_coordinates[i].data[0].item() * true_height)
                    coord_x = round(prev_coordinates[i].data[1].item() * true_width)
                    calculated_next_coordinate = get_next_coord((coord_y, coord_x, prev_moves[i]))
                    scaled_next_coordinate = (calculated_next_coordinate[0] / float(true_height), calculated_next_coordinate[1] / float(true_width))
                    next_q_value = robot_brain(next_game_board, torch.tensor(scaled_next_coordinate, device=device))
                    
                    #target = rewards[i][3] + robot_gamma * torch.max(next_q_value)
                    target = prev_q_value.clone()
                    target.data[prev_moves[i]] = rewards[i][3] + robot_gamma * torch.max(next_q_value)
                    target.detach()

                    robot_brain.zero_grad()
                    loss = functional.smooth_l1_loss(prev_q_value, target)
                    loss.backward()
                    for param in robot_brain.parameters(): param.data.sub_(param.grad.data * LEARNING_RATE_ROBOT * r)
                #     total_prev_q_value += prev_q_value[prev_moves[i]]
                #     total_target += target
                # loss = functional.smooth_l1_loss(total_prev_q_value, total_target)
                # robot_brain.zero_grad()
                # loss.backward()
                # for param in robot_brain.parameters(): param.data.sub_(param.grad.data * (LEARNING_RATE_ROBOT / r))
        
        def set_reward(self, prev_robot_transition_data, erase_me):
            robot_reward = 0
            if (prev_robot_transition_data[3], prev_robot_transition_data[4]) == (prev_robot_transition_data[0], prev_robot_transition_data[1]): # If haven't moved tiles
                robot_reward -= 0.5
            if raw_game_board_input[prev_robot_transition_data[3]][prev_robot_transition_data[4]][1] == -1: # Prev tile is now unowned
                robot_reward += 0.5
            elif raw_game_board_input[prev_robot_transition_data[3]][prev_robot_transition_data[4]][1] == 0: # Prev tile is now enemy's
                robot_reward -= 0.5
            if raw_game_board_input[prev_robot_transition_data[0]][prev_robot_transition_data[1]][1] == 1: # New tile is now ours
                if prev_robot_transition_data[2] == -1: # New tile used to be unowned
                    robot_reward += 1
                elif prev_robot_transition_data[2] == 0: # New tile used to be enemy's
                    robot_reward += 2
                else: # New tile used to be ours
                    robot_reward -= 0.25
            elif raw_game_board_input[prev_robot_transition_data[0]][prev_robot_transition_data[1]][1] == -1: # New tile is now unowned
                if prev_robot_transition_data[2] == -1: # New tile used to be unowned
                    robot_reward -= 0.5
                elif prev_robot_transition_data[2] == 0: # New tile used to be enemy's
                    robot_reward -= 0.5
                else: # New tile used to be ours
                    robot_reward -= 1
            elif raw_game_board_input[prev_robot_transition_data[0]][prev_robot_transition_data[1]][1] == 0: # New tile is now enemy's
                if prev_robot_transition_data[2] == -1: # New tile used to be unowned
                    robot_reward -= 1
                elif prev_robot_transition_data[2] == 0: # New tile used to be enemy's
                    robot_reward -= 0.5
                else: # New tile used to be ours
                    robot_reward += 0
            for robot_mapping in robot_rewards_sheet:
                if (robot_mapping[0], robot_mapping[1], robot_mapping[2]) == (prev_robot_transition_data[3], prev_robot_transition_data[4], prev_robot_transition_data[5]):
                    robot_mapping[3] = erase_me
                    #print("Rewards from prev turn: " + str(robot_reward), file=sys.stderr, flush=True)

    # Intent is a tuple of 3 values. [0] is y, [1] is x, [2] is direction
    def get_next_coord(intent):
        if intent[2] == 0: # UP
            next_coord = (max(0, intent[0] - 1), intent[1])
        elif intent[2] == 1: # LEFT
            next_coord = (intent[0], max(0, intent[1] - 1))
        elif intent[2] == 2: # RIGHT
            next_coord = (intent[0], min(true_width - 1, intent[1] + 1))
        elif intent[2] == 3: # DOWN
            next_coord = (min(true_height - 1, intent[0] + 1), intent[1])
        else: # STAY
            next_coord = (intent[0], intent[1])
        
        return next_coord

    def save_brains():
        torch.save(board_brain.state_dict(), "brains/HQ_brain" + file_suffix + ".pth")
        torch.save(robot_brain.state_dict(), "brains/robot_brain" + file_suffix + ".pth")
        reward_log = open("reward_log" + file_suffix + ".txt", "w")
        reward_log.write(str(curr_score) + "    \r\n")
        reward_log.close()

    board_brain = HQNeuralNetwork().to(device)
    robot_brain = RobotNeuralNetwork().to(device)
    if path.exists("brains/HQ_brain" + file_suffix + ".pth"):
        board_brain.load_state_dict(torch.load("brains/HQ_brain" + file_suffix + ".pth"))
    if path.exists("brains/robot_brain" + file_suffix + ".pth"):
        robot_brain.load_state_dict(torch.load("brains/robot_brain" + file_suffix + ".pth"))

    # game loop
    while True:
        my_matter, opp_matter = [int(i) for i in input().split()] if not debug else [50, 60]
        start_frame_time = time.time_ns()
        ideal_build_coords.clear()
        available_build_coords.clear()
        ideal_spawn_coords.clear()
        available_spawn_coords.clear()
        my_matter_input = torch.tensor([my_matter / float(150)], device=device, requires_grad=learning)
        opp_matter_input = torch.tensor([opp_matter / float(150)], device=device, requires_grad=learning)
        for i in range(true_height):
            for j in range(true_width):
                # owner: 1 = me, 0 = foe, -1 = neutral
                
                centered_i, centered_j = i + (HEIGHT - true_height) // 2, j + (WIDTH - true_width) // 2
                
                scrap_amount, owner, units, recycler, can_build, can_spawn, in_range_of_recycler = [int(k) for k in input().split()] if not debug else [-1, 0, 1, 2, 3, 4, 5]
                raw_game_board_input[i][j] = (scrap_amount, owner, units, recycler, can_build, can_spawn, in_range_of_recycler)

                if owner == 0: num_opp_tiles += 1
                elif owner == 1: 
                    if is_first_loop:
                        start_left = j <= float(true_width) / 2
                    num_my_tiles += 1
                    for k in range(units): my_robots_coords.append([i / float(true_height), j / float(true_width)])
                
                scaled_game_board_input[0][centered_i][centered_j] = scrap_amount / 10.0
                scaled_game_board_input[1][centered_i][centered_j] = owner
                scaled_game_board_input[2][centered_i][centered_j] = units / 10.0
                scaled_game_board_input[3][centered_i][centered_j] = recycler
        HQ_game_board_input = torch.tensor(scaled_game_board_input, device=device, requires_grad=learning)
        robot_game_board_input = torch.tensor(scaled_game_board_input, device=device, requires_grad=learning)

        furthest_x = -1
        for i in range(true_height):
            for j in range(true_width): # For calculating the spawn coordinates
                if (raw_game_board_input[i][min(true_width - 1, j + 1)][1] == -1 and raw_game_board_input[i][min(true_width - 1, j + 1)][0] > 0) or (raw_game_board_input[i][max(0, j - 1)][1] == -1 and raw_game_board_input[i][max(0, j - 1)][0] > 0) or (raw_game_board_input[min(true_height - 1, i + 1)][j][1] == -1 and raw_game_board_input[min(true_height - 1, i + 1)][j][0] > 0) or (raw_game_board_input[max(0, i - 1)][j][1] == -1 and raw_game_board_input[max(0, i - 1)][j][0] > 0):
                        if raw_game_board_input[i][j][5] == 1:
                            if start_left:
                                furthest_x = max(furthest_x, j)
                            else:
                                furthest_x = min(furthest_x, j)
                                if furthest_x == -1:
                                    furthest_x = j
        for i in range(true_height):
            for j in range(true_width):
                if raw_game_board_input[i][j][1] == 1:
                    prime_location = raw_game_board_input[i][min(true_width - 1, j + 1)][1] == 0 or raw_game_board_input[i][max(0, j - 1)][1] == 0 or raw_game_board_input[min(true_height - 1, i + 1)][j][1] == 0 or raw_game_board_input[max(0, i - 1)][j][1] == 0

                    if raw_game_board_input[i][j][4] == 1 and raw_game_board_input[i][j][2] == 0 and raw_game_board_input[i][j][6] == 0 and raw_game_board_input[i][min(true_width - 1, j + 1)][6] == 0 and raw_game_board_input[i][max(0, j - 1)][6] == 0 and raw_game_board_input[min(true_height - 1, i + 1)][j][6] == 0 and raw_game_board_input[max(0, i - 1)][j][6] == 0 and ((raw_game_board_input[i][min(true_width - 1, j + 1)][1] == -1 and raw_game_board_input[i][min(true_width - 1, j + 1)][0] > 0) or (raw_game_board_input[i][max(0, j - 1)][1] == -1 and raw_game_board_input[i][max(0, j - 1)][0] > 0) or (raw_game_board_input[min(true_height - 1, i + 1)][j][1] == -1 and raw_game_board_input[min(true_height - 1, i + 1)][j][0] > 0) or (raw_game_board_input[max(0, i - 1)][j][1] == -1 and raw_game_board_input[max(0, i - 1)][j][0] > 0)):
                        available_build_coords.append((i, j))
                    
                    if prime_location:
                        if raw_game_board_input[i][j][4] == 1 and raw_game_board_input[i][j][2] == 0:
                            ideal_build_coords.append((i, j))
                        if raw_game_board_input[i][j][5] == 1:
                            ideal_spawn_coords.append((i, j))
                    elif (raw_game_board_input[i][min(true_width - 1, j + 1)][1] == -1 and raw_game_board_input[i][min(true_width - 1, j + 1)][0] > 0) or (raw_game_board_input[i][max(0, j - 1)][1] == -1 and raw_game_board_input[i][max(0, j - 1)][0] > 0) or (raw_game_board_input[min(true_height - 1, i + 1)][j][1] == -1 and raw_game_board_input[min(true_height - 1, i + 1)][j][0] > 0) or (raw_game_board_input[max(0, i - 1)][j][1] == -1 and raw_game_board_input[max(0, i - 1)][j][0] > 0):
                        if raw_game_board_input[i][j][5] == 1:

                            if start_left:
                                if j + 2 >= furthest_x:
                                    available_spawn_coords.append((i, j))
                            else:
                                if j - 2 <= furthest_x:
                                    available_spawn_coords.append((i, j))
        end_frame_time = time.time_ns()

        hq_reward = prev_num_opp_tiles - num_opp_tiles + num_my_tiles - prev_num_my_tiles
        curr_score = num_my_tiles - num_opp_tiles


        for dest in prev_destination_owners:
            robot_brain.set_reward(dest, hq_reward)

        # Train HQ
        if not is_first_loop and learning:
            curr_q_value = board_brain(HQ_prev_state.requires_grad_(learning), prev_my_matter.requires_grad_(learning), prev_opp_matter.requires_grad_(learning))
            next_q_value = board_brain(HQ_game_board_input.clone().detach(), my_matter_input.clone().detach(), opp_matter_input.clone().detach())
            board_brain.train()
            board_brain.learn(hq_reward, prev_action, curr_q_value.requires_grad_(learning), next_q_value)
        end_HQ_backward_time = time.time_ns()
        final_command = ""

        board_brain.eval()
        board_brain_output = board_brain(HQ_game_board_input, my_matter_input, opp_matter_input)
        
        start_robot_backward_time = time.time_ns()
        if not is_first_loop and learning:
            robot_brain.train()
            robot_brain.learn(robot_prev_state.requires_grad_(learning), prev_coordinates, prev_moves, robot_rewards_sheet, robot_game_board_input.clone().detach())
        end_robot_backward_time = time.time_ns()
        prev_coordinates.clear()
        prev_moves.clear()
        robot_rewards_sheet.clear()
        prev_destination_owners.clear()

        # Forward propogate for all my robots and get their actions
        intent_dict = {}
        robot_brain.eval()
        for coord in my_robots_coords:
            coord_y = round(coord[0] * true_height)
            coord_x = round(coord[1] * true_width)
            tensor_coord = torch.tensor(coord, device=device, requires_grad=learning)
            prev_coordinates.append(tensor_coord)
            
            robot_brain_output = robot_brain(robot_game_board_input, tensor_coord)
            action_index = torch.multinomial(robot_brain_output, 1).item() if (not learning or r < EPSILON) else random.randint(0, 4) # Was argmax before
            prev_moves.append(action_index)
            robot_rewards_sheet.append([coord_y, coord_x, action_index, 0]) # 0 is the reward (to be calculated next frame)
            if (coord_y, coord_x, action_index) in intent_dict.keys():
                intent_dict[(coord_y, coord_x, action_index)] += 1
            else:
                intent_dict[(coord_y, coord_x, action_index)] = 1
        for intent in intent_dict.keys():
            destination = get_next_coord(intent)

            prev_destination_owners_item = (destination[0], destination[1], raw_game_board_input[destination[0]][destination[1]][1], intent[0], intent[1], intent[2])
            if prev_destination_owners.count(prev_destination_owners_item) < 1: # (destination[0] != intent[0] or destination[1] != intent[1]) and 
                prev_destination_owners.append(prev_destination_owners_item)
            
            if destination != (intent[0], intent[1]):
                final_command += "MOVE " + str(intent_dict[intent]) + " " + str(intent[1]) + " " + str(intent[0]) + " " + str(destination[1]) + " " + str(destination[0]) + ";"
        end_robot_forward_time = time.time_ns()

        # Get action
        action_index = torch.multinomial(board_brain_output, 1).item() if not learning or r < EPSILON else random.randint(0, 2)

        # Get amount
        amount = my_matter // 10

        # Get coordinates
        can_act = True
        x = 0
        y = 0
        if action_index == 1:
            #if len(ideal_build_coords) > 0:
            #    coordinate = random.choice(ideal_build_coords)
            #    x = coordinate[1]
            #    y = coordinate[0]
            if len(available_build_coords) > 0:
                coordinate = random.choice(available_build_coords)
                x = coordinate[1]
                y = coordinate[0]
            else:
                can_act = False
        elif action_index == 2:
            if len(available_spawn_coords) > 0 or len(ideal_spawn_coords) > 0:
                if len(ideal_spawn_coords) > 0:
                    #if random.getrandbits(1): # Add a bit of randomness to the aggression
                    coordinate = random.choice(ideal_spawn_coords)
                    x = coordinate[1]
                    y = coordinate[0]
                    #else:
                    #    coordinate = random.choice(available_spawn_coords)
                    #    x = coordinate[1]
                    #    y = coordinate[0]
                else:
                    coordinate = random.choice(available_spawn_coords)
                    x = coordinate[1]
                    y = coordinate[0]
            else:
                can_act = False

        if action_index == 0: # WAIT
            prev_action = 0
            final_command += "WAIT"
        elif action_index == 1: # BUILD
            prev_action = 1
            final_command += "BUILD " + str(x) + " " + str(y) + ";"
        else:
            prev_action = 2
            final_command += "SPAWN " + str(amount) + " " + str(x) + " " + str(y) + ";"
                
        HQ_prev_state = torch.clone(HQ_game_board_input)
        robot_prev_state = torch.clone(robot_game_board_input)
        prev_my_matter = torch.clone(my_matter_input)
        prev_opp_matter = torch.clone(opp_matter_input)
        prev_num_my_tiles = num_my_tiles
        prev_num_opp_tiles = num_opp_tiles
        num_my_tiles = 0
        num_opp_tiles = 0
        my_robots_coords.clear()

        print(final_command)

        start_save_time = time.time_ns()
        if not debug and learning: save_brains()

        print("Input reading time: " + str((end_frame_time - start_frame_time) / 1000000), file=sys.stderr, flush=True)
        print("HQ backward time: " + str((end_HQ_backward_time - end_frame_time) / 1000000), file=sys.stderr, flush=True)
        print("HQ forward time: " + str((start_robot_backward_time - end_HQ_backward_time) / 1000000), file=sys.stderr, flush=True)
        print("Robot backward time: " + str((end_robot_backward_time - start_robot_backward_time) / 1000000), file=sys.stderr, flush=True)
        print("Robot forward time: " + str((end_robot_forward_time - end_robot_backward_time) / 1000000), file=sys.stderr, flush=True)
        print("Save time: " + str((time.time_ns() - start_save_time) / 1000000), file=sys.stderr, flush=True)
        print("-------------------------", file=sys.stderr, flush=True)
        print("Total frame time: " + str((time.time_ns() - start_frame_time) / 1000000), file=sys.stderr, flush=True)
        print("", file=sys.stderr, flush=True)

        if debug and debug_itr >= NUM_DEBUG_ITR: quit()
        is_first_loop = False
        n += 1
        debug_itr += 1
