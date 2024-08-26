import os
import sys
import numpy as np
import argparse
from env import MyFrozenLake
from agent import Agent

# Get access to the HXP file
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from ESX import ESX
from ESX_tools import transition, terminal, complement, preprocess, readable_features

if __name__ == "__main__":

    #  Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-map', '--map_name', default="4x4", help="Map's dimension (nxn)", type=str, required=False)
    parser.add_argument('-policy', '--policy_name', default="4x4", help="Common part of policies name", type=str, required=False)
    parser.add_argument('-ep', '--nb_episodes', default=1, help="Number of episodes for a classic test of agent's policy", type=int, required=False)
    parser.add_argument('-k', '--length_k', default=5, help="Length of scenarios", type=int, required=False)
    parser.add_argument('-csv', '--csv_filename', default="scores.csv", help="csv file to store ESX logs", type=str, required=False)

    parser.add_argument('-ESX', '--ESX', dest="COMPUTE_ESX", action="store_true", help="Compute ESX", required=False)
    parser.add_argument('-no_ESX', '--no_ESX', action="store_false", dest="COMPUTE_ESX", help="Do not compute ESX", required=False)
    parser.set_defaults(COMPUTE_ESX=True)


    parser.add_argument('-equiprobable', '--equiprobable', dest="equiprobable", action="store_true", help="Equiprobable transitions", required=False)
    parser.add_argument('-no_equiprobable', '--no_equiprobable', action="store_false", dest="equiprobable", help="Equiprobable transitions", required=False)
    parser.set_defaults(equiprobable=False)

    parser.add_argument('-features', '--many_features', dest="many_features", action="store_true", help="Several features to define agent's state", required=False)
    parser.add_argument('-no_features', '--no_many_features', action="store_false", dest="many_features", help="Only one feature to define the agent's state", required=False)
    parser.set_defaults(many_features=True)

    parser.add_argument('-strat', '--strategy', default="exh", help="Exploration strategy for generating HXp", type=str, required=False)
    parser.add_argument('-strats', '--strategies', default="", help="Exploration strategies for similarity measures", type=str, required=False)

    parser.add_argument('-pred_size', '--pred_size', default='[2,2]', help="predicate size constraints for ESX (expected format: [int,int])",
                        type=str, required=False)
    parser.add_argument('-alpha', '--alpha', default='0.8', help="representative predicate threshold",
                        type=str, required=False)
    parser.add_argument('-sa', '--state_action', default='', help='Specific state-action couple to test (expected format: [int,int] or [(int,int,int,int,int),int]', type=str, required=False)

    parser.add_argument('-samples', '--complement_samples', default=0,
                        help="Number of state samples for the complement computation", type=int, required=False)


    args = parser.parse_args()

    # Get arguments
    MAP_NAME = args.map_name
    POLICY_NAME = args.policy_name
    EQUIPROBABLE = args.equiprobable
    K = args.length_k

    NUMBER_EPISODES = args.nb_episodes
    CSV_FILENAME = args.csv_filename
    STRATEGY = args.strategy
    COMPUTE_ESX = args.COMPUTE_ESX
    STRATEGIES = args.strategies
    FEATURES = args.many_features

    state_action = args.state_action
    pred_size = args.pred_size
    ALPHA = float(args.alpha)
    NB_SAMPLE = args.complement_samples

    # Path to obtain the Q table
    agent_Q_dirpath = "Q-tables" + os.sep + "Agent"

    # Path to store predicate trees and explanation
    if COMPUTE_ESX:
        esx_dirpath = 'Explanation' + os.sep + MAP_NAME
        if not os.path.exists(esx_dirpath):
            os.mkdir(esx_dirpath)
        esx_csv = esx_dirpath + os.sep + CSV_FILENAME
    else:
        esx_csv = 'trash.csv'

    # Convert state-action (type [int,int] or [tuple(int),int])
    if state_action:
        # type(state) = int
        if not FEATURES:
            split = state_action.split(',')
            STATE_ACTION = (int(split[0][1:]), int(split[1][:-1]))

        # type(state) = tuple(int)
        else:
            split = state_action.split('),')
            action = int(split[1][:-1])
            split_state = split[0][2:].split(',')
            state = tuple([int(elm) for elm in split_state])
            STATE_ACTION = (state, action)
    else:
        STATE_ACTION = None


    # Extract predicate size constraint
    split = pred_size.split(',')
    PRED_SIZE = [int(split[0][1:]), int(split[1][:-1])] # maximal number of terms, literals

    #  Envs initialisation
    if EQUIPROBABLE:
        env = MyFrozenLake(map_name=MAP_NAME, many_features=FEATURES)
    else:
        env = MyFrozenLake(map_name=MAP_NAME, slip_probas=[0.2, 0.6, 0.2], many_features=FEATURES)

    #  Agent initialization
    agent = Agent(POLICY_NAME, env)

    #  Load Q table
    agent.load(agent_Q_dirpath)

    #  Build feature domains dict
    if FEATURES:
        feat_domains = {i: [] for i in range(len(list(env.P.keys())[0]))}

        # position
        feat_domains[0] = [i for i in range(env.nRow*env.nCol)]

        # previous position
        # hole position
        previous_pos = []
        hole_pos = []
        for i in range(env.nRow):
            for j in range(env.nCol):
                if bytes(env.desc[i, j]) in b"H": # hole
                    hole_pos.append(env.to_s(i,j))
                else: # previous position
                    previous_pos.append(env.to_s(i,j))
        feat_domains[1] = previous_pos
        feat_domains[2] = hole_pos

        # manhattan distance
        feat_domains[3] = [i for i in range(env.nRow + env.nCol - 1)]

        # number of holes
        feat_domains[4] = [env.hole_cpt]

    # Initialize HXP class
    if COMPUTE_ESX and FEATURES:

        functions = [transition, terminal, complement, preprocess, readable_features]
        add_info = {}
        esx = ESX('FL', agent, env, K, feat_domains, PRED_SIZE, ALPHA, functions, add_info)

    # Display agent's policy

    current = None
    q_vals = set()
    for key, values in agent.Q.items():

        if current != key[0]:
            if current is None:
                current = key[0]
            else:
                print('From position {}, actions {}'.format(current, q_vals))
                print()
                q_vals = set()
                current = key[0]
        q_vals.add(np.argmax(values))

        print('From state {} -- Action {}'.format(key, np.argmax(values)))
        print("-------------------------------------")

    print('From position {}, actions {}'.format(current, q_vals))

    # Compute ESX for a specific state-action couple
    if STATE_ACTION and COMPUTE_ESX:

        if not STRATEGIES:
            esx.explain('no_user', STATE_ACTION, [STRATEGY], NB_SAMPLE, esx_csv)
        else:
            esx.explain('compare', STATE_ACTION, STRATEGIES, NB_SAMPLE, esx_csv)

    # Classic testing loop
    else:
        sum_reward = 0
        misses = 0
        steps_list = []
        nb_episode = NUMBER_EPISODES

        for episode in range(1, nb_episode + 1):
            obs = env.reset()
            done = False
            score = 0
            steps = 0
            while not done:

                steps += 1
                env.render()
                action = agent.predict(obs)
                print('Action: {}'.format(['Left', 'Down', 'Right', 'Up'][action]))

                #  Compute ESX
                if COMPUTE_ESX and FEATURES:
                    esx.explain('user', (obs, action), [STRATEGY], NB_SAMPLE, esx_csv)

                obs, reward, done, info = env.step(action)
                score += reward

                # Store infos
                if done and reward == 1:
                    steps_list.append(steps)
                elif done and reward == 0:
                    misses += 1

            sum_reward += score
            print('Episode:{} Score: {}'.format(episode, score))

        if nb_episode > 1:
            print('Score: {}'.format(sum_reward/nb_episode))
            print('----------------------------------------------')
            print('Average of {:.0f} steps to reach the goal position'.format(np.mean(steps_list)))
            print('Fall {:.2f} % of the times'.format((misses / nb_episode) * 100))
            print('----------------------------------------------')
