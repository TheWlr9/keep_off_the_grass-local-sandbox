import getopt
import sys
import numpy
import os
from os import path
import pickle
import random
import time
import torch
from torch import nn
from torch.nn import functional
from contextlib import redirect_stderr

NUM_DEBUG_ITR = 2
debug_itr = 1

WIDTH, HEIGHT = 24, 13

HQ_SAMPLE_SIZE = 200
ROBOT_SAMPLE_SIZE = 1000
EPSILON = 0.5
LEARNING_RATE_HQ = 0.1
LEARNING_RATE_ROBOT = 0.001

robot_gamma = 0.5
HQ_gamma = 0.9
r = 1.0 # Learning rate multiplier
device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
file_suffix = "_v0"
debug = False
learning = False # If True, brains will sometimes take random choices based on 1 - EPSILON probability
train_only = False # If True, will only train from specific files and exit. Will not test/play game



# This code will be run upon startup
def input_help():
    print(file_suffix, file=sys.stderr, flush=True)
    print("Usage: ai.py -f <file_suffix> -r <learning_rate_weight> [-l] [-d] [-t]", file=sys.stderr, flush=True)
    sys.exit(2)

try:
    opts, args = getopt.getopt(sys.argv[1:], "hltr:f:d")
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
    if opt == "-t":
        train_only = True

with open("ai" + file_suffix + ".log", "w") as stderr, redirect_stderr(stderr):

    # NOTE game_board_inputs DOES NOT have gradiants for a REASON!!!
    HQ_game_board_input = torch.zeros(4, HEIGHT, WIDTH, device=device)
    robot_game_board_input = torch.zeros(4, HEIGHT, WIDTH, device=device)
    my_matter_input = torch.zeros(1, device=device, requires_grad=learning)
    opp_matter_input = torch.zeros(1, device=device, requires_grad=learning)
    true_width, true_height = [int(i) for i in input().split()] if not debug and not train_only else [21, 10]

    # prev_HQ_transitions element contains data in the following format: [prev_game_board, prev_my_matter, prev_opp_matter, action, reward, next_game_board, next_my_matter, next_opp_matter]
    prev_HQ_transition = []
    # prev_robot_transitions element contains data in the following format: [prev_game_board, prev_coords, action_index, reward, next_game_board]
    prev_robot_transitions = []

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
    start_left = True # Start on the left side of the board?

    is_first_loop = True
    n = 1 # Num frames elapsed

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
        
        def learn(self, prev_transition):
            curr_q_value = board_brain(prev_transition[0].requires_grad_(True), prev_transition[1].requires_grad_(True), prev_transition[2].requires_grad_(True))
            target = curr_q_value.clone()
            target.data[prev_transition[3].item()] = prev_transition[4].item() + HQ_gamma * torch.max(board_brain(prev_transition[5], prev_transition[6], prev_transition[7])) # These tensors for the input in the forward pass are already detached from the gradient graph
            target.detach()
            
            board_brain.zero_grad()
            loss = functional.smooth_l1_loss(curr_q_value, target)
            
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
        
        def learn(self, prev_transition): # action_index (prev_transition[2]) and reward (prev_transition[3]) are not tensors (just because lazy). This is unlike the prev_transition parameter for the HQNeuralNetwork
            prev_q_value = robot_brain(prev_transition[0].requires_grad_(True), prev_transition[1].requires_grad_(True))
            coord_y = round(prev_transition[1].data[0].item() * true_height)
            coord_x = round(prev_transition[1].data[1].item() * true_width)
            calculated_next_coordinate = get_next_coord((coord_y, coord_x, prev_transition[2]))
            scaled_next_coordinate = (calculated_next_coordinate[0] / float(true_height), calculated_next_coordinate[1] / float(true_width))
            next_q_value = robot_brain(prev_transition[4], torch.tensor(scaled_next_coordinate, device=device))
            
            target = prev_q_value.clone()
            target.data[prev_transition[2]] = prev_transition[3] + robot_gamma * torch.max(next_q_value)
            target.detach()

            robot_brain.zero_grad()
            loss = functional.smooth_l1_loss(prev_q_value, target)
            loss.backward()
            for param in robot_brain.parameters(): param.data.sub_(param.grad.data * LEARNING_RATE_ROBOT * r)

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

    def save_transitions(sum_file_name, transitions_file_name, transitions):
        # Create the files first
        if not path.exists(sum_file_name): 
            file = open(sum_file_name, "x")
            file.close()
        if not path.exists(transitions_file_name): 
            file = open(transitions_file_name, "x")
            file.close()
        
        file = open(sum_file_name, "r+")
        num = file.readline()
        num = len(transitions) if num == "" else int(num) + len(transitions)
        file.seek(0)
        file.truncate()
        file.write(str(num) + "     \r\n")
        file.close()
        
        file = open(transitions_file_name, "ab")
        for transition in transitions:
            pickle.dump(transition, file)
        file.close()

    def save_brains():
        torch.save(board_brain.state_dict(), "brains/HQ_brain" + file_suffix + ".pth")
        torch.save(robot_brain.state_dict(), "brains/robot_brain" + file_suffix + ".pth")
        reward_log = open("reward_log" + file_suffix + ".txt", "w")
        reward_log.truncate()
        reward_log.write(str(curr_score) + "    \r\n")
        reward_log.close()

    board_brain = HQNeuralNetwork().to(device)
    robot_brain = RobotNeuralNetwork().to(device)
    if path.exists("brains/HQ_brain" + file_suffix + ".pth"):
        board_brain.load_state_dict(torch.load("brains/HQ_brain" + file_suffix + ".pth"))
    if path.exists("brains/robot_brain" + file_suffix + ".pth"):
        robot_brain.load_state_dict(torch.load("brains/robot_brain" + file_suffix + ".pth"))

    if train_only:
        start_training_time = time.time_ns()
        board_brain.train()
        robot_brain.train()
        
        # Train HQ
        transitions_buffer = []
        total_transitions_file = open("transitions/HQ_sum_transitions" + file_suffix + ".txt", "r")
        num_transitions = int(total_transitions_file.readline())
        total_transitions_file.close()
        transitions_file = open("transitions/HQ_transitions" + file_suffix + ".tns", "rb")
        start_reading_HQ_file = time.time_ns()
        for i in range(num_transitions):
            transitions_buffer.append([])
            transitions_buffer[i].append(pickle.load(transitions_file))
        end_reading_HQ_file = time.time_ns()
        transitions_file.close()
        random_indices = torch.randint(0, num_transitions, (HQ_SAMPLE_SIZE,))
        start_HQ_backprop = time.time_ns()
        for index in random_indices.data:
            board_brain.learn(transitions_buffer[index][0])
        end_HQ_backprop = time.time_ns()
        
        transitions_buffer.clear()
        
        # Train Robot
        total_transitions_file = open("transitions/robot_sum_transitions" + file_suffix + ".txt", "r")
        num_transitions = int(total_transitions_file.readline())
        total_transitions_file.close()
        transitions_file = open("transitions/robot_transitions" + file_suffix + ".tns", "rb")
        start_reading_robot_file = time.time_ns()
        for i in range(num_transitions):
            transitions_buffer.append([])
            transitions_buffer[i].append(pickle.load(transitions_file))
        end_reading_robot_file = time.time_ns()
        transitions_file.close()
        random_indices = torch.randint(0, num_transitions, (ROBOT_SAMPLE_SIZE,))
        start_robot_backprop = time.time_ns()
        for index in random_indices.data:
            robot_brain.learn(transitions_buffer[index][0])
        end_robot_backprop = time.time_ns()
        
        start_saving_brains = time.time_ns()
        torch.save(board_brain.state_dict(), "brains/HQ_brain" + file_suffix + ".pth")
        torch.save(robot_brain.state_dict(), "brains/robot_brain" + file_suffix + ".pth")
        end_saving_brains = time.time_ns()
        
        print("\tReading HQ transitions time: \t" + str((end_reading_HQ_file - start_reading_HQ_file) / 1000000) + "ms")
        print("\tHQ learning time: \t\t" + str((end_HQ_backprop - start_HQ_backprop) / 1000000) + "ms")
        print("\tReading robot transitions time: " + str((end_reading_robot_file - start_reading_robot_file) / 1000000) + "ms")
        print("\tRobot learning time: \t\t" + str((end_robot_backprop - start_robot_backprop) / 1000000) + "ms")
        print("\tSaving brains time: \t\t" + str((end_saving_brains - start_saving_brains) / 1000000) + "ms")
        print("\t-----")
        print("\tTotal batch training time: \t" + str((end_saving_brains - start_training_time) / 1000000) + "ms")
        
        os.remove("transitions/HQ_sum_transitions" + file_suffix + ".txt")
        os.remove("transitions/HQ_transitions" + file_suffix + ".tns")
        os.remove("transitions/robot_sum_transitions" + file_suffix + ".txt")
        os.remove("transitions/robot_transitions" + file_suffix + ".tns")
    else:
        board_brain.eval()
        robot_brain.eval()

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
            
            # Two separate copies of the same input because we don't want the gradients of the two tensors to conflict with each other
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

            # Calculate reward
            hq_reward = prev_num_opp_tiles - num_opp_tiles + num_my_tiles - prev_num_my_tiles
            curr_score = num_my_tiles - num_opp_tiles
            clone_of_robot_game_board_input = robot_game_board_input.clone().detach()
            for k in range(len(prev_robot_transitions)):
                prev_robot_transitions[k].append(hq_reward)
                prev_robot_transitions[k].append(clone_of_robot_game_board_input)

            # Train HQ
            prev_HQ_transition.append(torch.tensor(hq_reward, device=device))
            if not is_first_loop and learning:
                prev_HQ_transition.append(HQ_game_board_input.clone().detach())
                prev_HQ_transition.append(my_matter_input.clone().detach())
                prev_HQ_transition.append(opp_matter_input.clone().detach())
                #board_brain.learn(prev_HQ_transition)
            end_HQ_backward_time = time.time_ns()
            if not is_first_loop and learning:
                save_transitions("transitions/HQ_sum_transitions" + file_suffix + ".txt", "transitions/HQ_transitions" + file_suffix + ".tns", [prev_HQ_transition])
                save_transitions("transitions/robot_sum_transitions" + file_suffix + ".txt", "transitions/robot_transitions" + file_suffix + ".tns", prev_robot_transitions)
            
            final_command = ""

            board_brain_output = board_brain(HQ_game_board_input, my_matter_input, opp_matter_input)
            
            start_robot_backward_time = time.time_ns()
            #if not is_first_loop and learning:
            #    for transition in prev_robot_transitions:
            #        robot_brain.learn(transition)
            end_robot_backward_time = time.time_ns()

            # Forward propogate for all my robots and get their actions
            intent_dict = {}
            robot_frame_id = 0
            prev_robot_transitions.clear()
            for coord in my_robots_coords:
                coord_y = round(coord[0] * true_height)
                coord_x = round(coord[1] * true_width)
                tensor_coord = torch.tensor(coord, device=device, requires_grad=learning)
                
                # Append the current game state to the list of robot transitions (important for experience replay)
                prev_robot_transitions.append([])
                prev_robot_transitions[robot_frame_id].append(robot_game_board_input)
                prev_robot_transitions[robot_frame_id].append(tensor_coord)
                
                robot_brain_output = robot_brain(robot_game_board_input, tensor_coord)
                if robot_frame_id == 0:
                    print(str(robot_brain_output), file=sys.stderr, flush=True)
                
                action_index = torch.multinomial(robot_brain_output, 1).item() if (not learning or r < EPSILON) else random.randint(0, 4)
                prev_robot_transitions[robot_frame_id].append(action_index)
                
                if (coord_y, coord_x, action_index) in intent_dict.keys():
                    intent_dict[(coord_y, coord_x, action_index)] += 1
                else:
                    intent_dict[(coord_y, coord_x, action_index)] = 1
                
                robot_frame_id += 1
            for intent in intent_dict.keys():
                # Use this to conglomerate similar robots actions and coordinates into one command
                destination = get_next_coord(intent)
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

            chosen_HQ_action = 0
            if action_index == 0: # WAIT
                chosen_HQ_action = 0
                final_command += "WAIT"
            elif action_index == 1: # BUILD
                chosen_HQ_action = 1
                final_command += "BUILD " + str(x) + " " + str(y) + ";"
            else:
                chosen_HQ_action = 2
                final_command += "SPAWN " + str(amount) + " " + str(x) + " " + str(y) + ";"
            
            # Save the transitions and misc data from this frame
            prev_HQ_transition.clear()
            prev_HQ_transition.append(torch.clone(HQ_game_board_input))
            prev_HQ_transition.append(torch.clone(my_matter_input))
            prev_HQ_transition.append(torch.clone(opp_matter_input))
            prev_HQ_transition.append(torch.tensor(chosen_HQ_action, device=device))
            prev_num_my_tiles = num_my_tiles
            prev_num_opp_tiles = num_opp_tiles
            num_my_tiles = 0
            num_opp_tiles = 0
            my_robots_coords.clear()

            print(final_command) # Actually output the command!

            start_save_time = time.time_ns()
            #if not debug and learning: save_brains()
            reward_log = open("reward_log" + file_suffix + ".txt", "w")
            reward_log.write(str(curr_score) + "    \r\n")
            reward_log.close()

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
