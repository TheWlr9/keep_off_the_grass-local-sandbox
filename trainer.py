import datetime
import os
from os import path
import sys
import getopt
import math
import subprocess
import time

BATCH_SIZE = 10

game_sim_cmd = ["java", "--add-opens", "java.base/java.lang=ALL-UNNAMED", "-Dnotimeout=\"true\"", "-jar", "ea-2022-keep-off-the-grass-1.0-SNAPSHOT.jar", "\"python", "Boss-easy.pyc\"", "local"]

def help():
    print("Usage: trainer.py (-s <num_simulations> | -t <time_of_training>) -f <file_suffix> -i <ai_python_script> [-d]")
    sys.exit(2)

def train(debug, ai_script, num_simulations, num_seconds, file_suffix):
    print("AI TRAINER STARTED")
    print("")

    global BATCH_SIZE
    _num_simulations = int(num_simulations)
    _num_seconds = int(num_seconds)

    training_command = ["python", ai_script, "-t", "-f", file_suffix]
    start_seconds = time.time()
    list_times = []
    total_avg_reward = 0
    batch_avg_reward = 0
    n = 0
    first_batch_score = 0
    last_batch_score = 0

    print("BATCH 0")
    print("---------")
    if _num_simulations > 0:
        for i in range(_num_simulations):
            r = 1.0 - math.sqrt(float(i)) / math.sqrt(float(_num_simulations))
            command_words = ["\"python", ai_script]
            if debug: command_words.append("-d")
            command_words.append("-l")
            command_words.append("-r")
            command_words.append(str(r))
            command_words.append("-f")
            command_words.append(file_suffix + "\"")

            local_game_sim_cmd = game_sim_cmd.copy()
            for j in range(len(command_words)):
                local_game_sim_cmd.insert(6 + j, command_words[j])
            
            #run simulation
            print("\tTesting/playing...", end='\r')
            list_time_start = time.time()
            simulation_output = subprocess.run(local_game_sim_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            list_times.append(time.time() - list_time_start)
            reward_file = open("reward_log" + file_suffix + ".txt", "r")
            reward = int(reward_file.readline())
            reward_file.close()
            batch_avg_reward += reward
            total_avg_reward += reward

            if simulation_output.returncode != 0:
                print("")
                print("Something errored during testing/playing!")
                print(simulation_output.stderr)
                sys.exit(1)
            elif i % BATCH_SIZE == BATCH_SIZE - 1 and i > 0: # Gather data
                batch_avg_reward /= BATCH_SIZE
                if i == BATCH_SIZE - 1: first_batch_score = batch_avg_reward
                last_batch_score = batch_avg_reward
                avg_time = 0
                for curr_time in list_times:
                    avg_time += curr_time
                avg_time /= len(list_times)
                
                # Train the batch
                print("\tTesting/playing... \t\tDone!")
                print("\tTraining...", end='\r')
                start_training_time = time.time()
                simulation_output = subprocess.run(training_command, text=True)
                if simulation_output.returncode != 0:
                    print("")
                    print("Something errored during training!")
                    print(simulation_output.stderr)
                    sys.exit(1)
                print("\tTraining... \t\t\tDone!")
                
                # Print out info
                print("\n\tStats for simulations " + str(i - (BATCH_SIZE - 1)) + "-" + str(i))
                print("\t--------------------------------------------")
                print("\tLatest learning rate scalar: \t" + str(r))
                print("\tBatch avg score: \t\t" + str(batch_avg_reward))
                print("\tElapsed time: \t\t\t" + str(datetime.timedelta(seconds=time.time() - start_seconds)))
                print("\tEstimated time remaining: \t" + str(datetime.timedelta(seconds=(_num_simulations - (i + 1)) * (avg_time + time.time() - start_training_time))))
                print("\t--------------------------------------------")
                print("\t")
                if i <= _num_simulations - BATCH_SIZE:
                    print("BATCH " + str((i + 1) // BATCH_SIZE))
                    print("---------")

                list_times.clear()
                batch_avg_reward = 0
    else:
        end_seconds = start_seconds + _num_seconds
        while time.time() < end_seconds:
            r = 1.0 - math.sqrt(float(time.time() - start_seconds)) / math.sqrt(float(_num_seconds))
            command_words = ["\"python", ai_script]
            if debug: command_words.append("-d")
            command_words.append("-l")
            command_words.append("-r")
            command_words.append(str(r))
            command_words.append("-f")
            command_words.append(file_suffix + "\"")

            local_game_sim_cmd = game_sim_cmd.copy()
            for i in range(len(command_words)):
                local_game_sim_cmd.insert(6 + i, command_words[i])
            
            #run simulation
            print("\tTesting/playing...", end='\r')
            simulation_output = subprocess.run(local_game_sim_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            reward_file = open("reward_log" + file_suffix + ".txt", "r")
            reward = int(reward_file.readline())
            reward_file.close()
            batch_avg_reward += reward
            total_avg_reward += reward

            if simulation_output.returncode != 0:
                print("")
                print("Something errored during testing/playing!")
                print(simulation_output.stderr)
                sys.exit(1)
            elif n % BATCH_SIZE == BATCH_SIZE - 1 and n > 0: # Gather data
                batch_avg_reward /= BATCH_SIZE
                if n == BATCH_SIZE - 1: first_batch_score = batch_avg_reward
                last_batch_score = batch_avg_reward
                
                # Train the batch
                print("\tTesting/playing... \t\tDone!")
                print("\tTraining...", end='\r')
                simulation_output = subprocess.run(training_command, text=True)
                if simulation_output.returncode != 0:
                    print("")
                    print("Something errored during training!")
                    print(simulation_output.stderr)
                    sys.exit(1)
                print("\tTraining... \t\t\tDone!")

                # Print out info
                print("\n\tStats for simulations " + str(n - (BATCH_SIZE - 1)) + "-" + str(n))
                print("\t--------------------------------------------")
                print("\tLatest learning rate scalar: \t" + str(r))
                print("\tBatch avg score: \t\t" + str(batch_avg_reward))
                print("\tElapsed time: \t\t\t" + str(datetime.timedelta(seconds=time.time() - start_seconds)))
                print("\tTime remaining: \t\t" + str(datetime.timedelta(seconds=end_seconds - time.time())))
                print("\t--------------------------------------------")
                print("\t")
                if time.time() < end_seconds - 5:
                    print("BATCH " + str((n + 1) // BATCH_SIZE))
                    print("---------")

                batch_avg_reward = 0

            n += 1

    os.remove("reward_log" + file_suffix + ".txt")
    
    print("\n ________________________________")
    print("|\tAI TRAINER FINISHED \t |")
    print("|Time taken: \t   " + str(datetime.timedelta(seconds=time.time() - start_seconds)) + "|")
    print("|# of simulations: \t\t" + str(max(_num_simulations, n)) + "|")
    print("|Delta avg score: \t\t" + str(last_batch_score - first_batch_score) + "|")
    print("|Avg score: \t\t     " + str(float(total_avg_reward) / max(n, _num_simulations)) + "|")
    print(" --------------------------------")
    
    if path.exists("transitions/HQ_sum_transitions" + file_suffix + ".txt"):
        os.remove("transitions/HQ_sum_transitions" + file_suffix + ".txt")
    if path.exists("transitions/HQ_transitions" + file_suffix + ".tns"):
        os.remove("transitions/HQ_transitions" + file_suffix + ".tns")
    if path.exists("transitions/robot_sum_transitions" + file_suffix + ".txt"):
        os.remove("transitions/robot_sum_transitions" + file_suffix + ".txt")
    if path.exists("transitions/robot_transitions" + file_suffix + ".tns"):
        os.remove("transitions/robot_transitions" + file_suffix + ".tns")

def main(argv):
    opts = []
    try:
        opts, args = getopt.getopt(argv, "hs:t:f:i:d")
    except getopt.GetoptError:
        help()

    file_suffix = ""
    debug_mode = False
    ai_script = ""
    num_simulations = 0
    total_seconds = 0
    for opt, arg in opts:
        if opt == "-f":
            file_suffix = arg
        if opt == "-h":
            help()
        if opt == "-d":
            debug_mode = True
        if opt == "-i":
            ai_script = arg
        if opt == "-s":
            num_simulations = arg
        elif opt == "-t":
            total_seconds = arg
    
    train(debug_mode, ai_script, num_simulations, total_seconds, file_suffix)


if __name__ == "__main__":
    main(sys.argv[1:])