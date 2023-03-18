import sys
import numpy as np
import math
import random

import json
import requests

import gym
import gym_maze
from gym_maze.envs.maze_manager import MazeManager
from riddle_solvers import *



def number_to_move(number):
    moves = {1: "N", 2: "E", 4: "S", 8: "W"}
    return moves[number]


def possactions(agentloc):
    possible_actions = []
    agentmove = sample_maze[agentloc[1]][agentloc[0]]
    #print(sample_maze)
    possible_actions.append(number_to_move(agentmove))
    # rows
    # if right
    if agentloc[1] != 9:
        agentmove = sample_maze[agentloc[1]+1][agentloc[0]]
        if agentmove == 8:
            possible_actions.append(number_to_move(2))
    # if left
    if agentloc[1] != 0:
        agentmove = sample_maze[agentloc[1]-1][agentloc[0]]
        if agentmove == 2:
            possible_actions.append(number_to_move(8))
    # columns
    # if up
    if agentloc[0] != 0:
        agentmove = sample_maze[agentloc[1]][agentloc[0]-1]
        if agentmove == 4:
            possible_actions.append(number_to_move(1))
    # if down
    if agentloc[0] != 9:
        agentmove = sample_maze[agentloc[1]][agentloc[0]+1]
        if agentmove == 1:
            possible_actions.append(number_to_move(4))
    return possible_actions


def select_action(state, q_table):
    actions = ['W', 'S', 'E', 'N']
    action_index = np.argmax(q_table[state[0], state[1]])
    random_action = actions[action_index]
    return random_action, action_index


def getnextstate(action, location):
    if action == "E":
        location = (location[0], location[1] + 1)
    if action == "W":
        location = (location[0], location[1] - 1)
    if action == "S":
        location = (location[0] + 1, location[1])
    if action == "N":
        location = (location[0] - 1, location[1])
    return location


def actionindex(action):
    if action == "W":
        action_index = 0
    elif action == "S":
        action_index = 1
    elif action == "E":
        action_index = 2
    else:
        action_index = 3
    return action_index


def choose_action(q_table, state, actions, epsilon):
    state = state[0][::-1]
    actions_indices = ["W", "S", "E", "N"]
    if np.random.uniform(0, 1) < epsilon:
        # Choose a random action with probability epsilon
        action = np.random.choice(actions)
        action_idx = actionindex(action)
    else:
        # Choose the action with the highest Q-value for the current state
        action_idx = np.argmax(q_table[state[0], state[1]])
        action = actions_indices[action_idx]
    return action, action_idx

# Define a function to update the Q-values based on a single experience
def update_q_value(state, action, reward, next_state, q_table, learning_rate):
    # Set the learning parameters
    discount_factor = 0.95
    actions = ["W", "S", "E", "N"]
    if next_state is None:
        q_table[state[0][1], state[0][0]][actions.index(action)] = reward
        return q_table
    # Find the maximum Q-value for the next state
    max_q_value = np.max(q_table[(next_state[0], next_state[1])])
    # Update the Q-value for the current state and action
    state = state[0][::-1]
    current_q_value = q_table[state[0], state[1]][actions.index(action)]
    new_q_value = current_q_value + learning_rate * (reward + discount_factor * max_q_value - current_q_value)
    #print(state, next_state)
    #print(q_table[state], current_q_value, max_q_value, new_q_value)
    q_table[state[0], state[1]][actions.index(action)] = new_q_value
    #print(q_table[state[0], state[1]], reward, current_q_value, max_q_value, new_q_value)
    return q_table


# def learning_rate_decay(episode, learning_rate):
#     return learning_rate * (0.1 ** (episode // 10))
#
# # Define function for epsilon decay
# def epsilon_decay(episode, epsilon):
#     return epsilon * (0.1 ** (episode // 10))

def get_reward(action, state, next_state, visited, captured, action_space):


    if next_state is None:
        return -10.0, visited, captured
    if len(action_space) == 1 and next_state[0] in visited:
        reward = -10.0
        return reward, visited, captured

    #for every move
    reward = -0.1

    if next_state[0] in visited:
        reward -= 0.1

    if next_state[0] not in visited:
        reward += 0.5

    if state[0] not in visited:
        visited.add(state[0])

    #captured a flag
    if [0, 0] in next_state[2] and 0 in next_state[1]:
        r_id = next_state[2].index([0, 0])
        r_id_1 = next_state[1].index(0)
        if r_id == r_id_1:
            if r_id not in captured:
                captured.add(r_id)
                reward += 0.7
            # else:
            #     reward -= 0.2

    if action == "S" and (state[0][1], state[0][0]) == (8, 9) or action == "E" and (state[0][1], state[0][0]) == (9, 8):
        # if len(captured) == 4:
        reward = 10
        # else:
        #     reward = -0.2
        return reward, visited, captured

    if len(captured) == 4:
        #if he gets closer to goal
        if( math.sqrt((next_state[0][0]-9)**2 + (next_state[0][1]-9)**2) < math.sqrt((state[0][1]-9)**2 + (state[0][1]-9)**2)):
            reward += 0.7
        # if he gets away of goal
        if( math.sqrt((next_state[0][0]-9)**2 + (next_state[0][1]-9)**2) > math.sqrt((state[0][1]-9)**2 + (state[0][1]-9)**2)):
            reward -= 0.1

    # # if he get close to childern
    if len(captured) != 4:
        if state[0] not in visited:
            index = np.argmin(state[1])
            if index not in captured:
                if next_state[1][index] < state[1][index]:
                    reward += 0.8
                else:
                    reward -= 0.5
        # for i in range(0, len(state[1])):
        #     if next_state[1][i] < state[1][i]:
        #         reward += 0.2
        #     elif next_state[1][i] > state[1][i]:
        #         reward -= 0.1
    #print(state, reward)
    return reward, visited, captured


def train(q_table):

    actions = ["W", "S", "E", "N"]
    # Run the Q-learning algorithm for a fixed number of episodes
    num_episodes = 100
    max_epsilon = 2.0
    learning_rate = 0.1
    epsilon_min = 0.01
    decay_rate = 0.001
    learning_rate_min = 0.01
    for episode in range(num_episodes):
        captured = set()
        visited = set()
        visited.add((0, 0))
        print(episode)
        # Initialize the state to the starting position
        state = manager.reset(agent_id)
        # epsilon = max(epsilon_min, epsilon_decay(episode, epsilon) * epsilon)
        epsilon = epsilon_min + (max_epsilon - epsilon_min) * np.exp(-decay_rate * episode)
        # Update the learning rate using a decay schedule
        # learning_rate = max(learning_rate_min, learning_rate_decay(episode, learning_rate) * learning_rate
        learning_rate = 1.0 / (1.0 + episode)
        sorted_distances = sorted(state[1])
        closest_child = state[1].index(sorted_distances[0])
        while True:
            if closest_child in captured:
                sorted_distances = sorted(state[1])
                closest_child = state[1].index(sorted_distances[0])

            current_state = state
            # Choose an action based on the current state
            action_space = possactions(state[0][::-1])
            #action, action_index = choose_action(q_table, state, possactions(state[0][::-1]), epsilon)
            action, action_index = choose_action(q_table, state, actions, epsilon)
            next_state = getnextstate(action, state[0][::-1])
            #print(closest_child)
            if action not in action_space:
                reward, visited, captured = get_reward(action, current_state, None, visited, captured,
                                                       action_space)
                q_table = update_q_value(state, action, reward, None, q_table, learning_rate)
                continue

            if next_state[1] < 0 or next_state[1] > 9 or next_state[0] < 0 or next_state[0] > 9:
                continue
            i = current_state[0][0]
            j = current_state[0][1]
            obv, reward, terminated, truncated, info = manager.step(agent_id, action)
            current_state[0] = (i, j)
            actual_next_state = obv
            actual_next_state[0] = next_state[::-1]
            reward, visited, captured = get_reward(action, current_state, actual_next_state, visited, captured, action_space)

            # Update the Q-value for the current state and action
            q_table = update_q_value(state, action, reward, next_state, q_table, learning_rate)

            # Check if the agent has reached the goal
            state = actual_next_state
            if next_state == (9, 9):
                print("reached")
                break
        print(captured)
    return q_table

def new_get_q_table():

    # Define the possible actions
    actions = ['W', 'S', 'E', 'N']

    # Initialize the Q-table with zeros
    q_table = np.full((10, 10, len(actions)), 0.0)
    train(q_table)

    # Define a function to choose an action using an epsilon-greedy policy
    return q_table




def local_inference(riddle_solvers):
    q_table = new_get_q_table()
    print(q_table)
    obv = manager.reset(agent_id)
    state_0 = obv
    for t in range(MAX_T):

        #action, action_index = select_action( (state_0[0]) )
        action, action_index = select_action( state_0[0][::-1], q_table)
        obv, reward, terminated, truncated, info = manager.step(agent_id, action)
        if not info['riddle_type'] == None:
            solution = riddle_solvers[info['riddle_type']](info['riddle_question'])
            obv, reward, terminated, truncated, info = manager.solve_riddle(info['riddle_type'], agent_id, solution)

        # THIS IS A SAMPLE TERMINATING CONDITION WHEN THE AGENT REACHES THE EXIT
        # IMPLEMENT YOUR OWN TERMINATING CONDITION
        if np.array_equal(obv[0], (len(sample_maze)-1, len(sample_maze[0])-1)):
            manager.set_done(agent_id)
            break  # Stop Agent

        if RENDER_MAZE:
            manager.render(agent_id)
        print(t)
        states[t] = [obv[0].tolist(), action_index, str(manager.get_rescue_items_status(agent_id))]


if __name__ == "__main__":
    sample_maze = np.load("hackathon_sample.npy")
    agent_id = "9"  # add your agent id here
    manager = MazeManager()
    manager.init_maze(agent_id, maze_cells=sample_maze)
    env = manager.maze_map[agent_id]
    riddle_solvers = {'cipher': cipher_solver, 'captcha': captcha_solver, 'pcap': pcap_solver, 'server': server_solver}
    maze = {}
    states = {}

    maze['maze'] = env.maze_view.maze.maze_cells.tolist()
    maze['rescue_items'] = list(manager.rescue_items_dict.keys())

    MAX_T = 5000
    RENDER_MAZE = True

    local_inference(riddle_solvers)
    with open("./states.json", "w") as file:
        json.dump(states, file)

    with open("./maze.json", "w") as file:
        json.dump(maze, file)