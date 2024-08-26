import argparse
import os
import csv

from ESX import ESX

# FL import
from FrozenLake.ESX_tools import transition as FL_transition
from FrozenLake.ESX_tools import terminal as FL_terminal
from FrozenLake.ESX_tools import complement as FL_complement
from FrozenLake.ESX_tools import preprocess as FL_preprocess
from FrozenLake.ESX_tools import readable_features as FL_readable_features
from FrozenLake.env import MyFrozenLake
from FrozenLake.agent import Agent

#  Convert a state-action couple from str to actual type
#  Input: environment name (str), state-action couple (str)
#  Output: state-action couple
def from_str_to_sa(env_name, str_sa):
    if env_name == 'FL':
        split = str_sa.split('),')
        action = int(split[1][:-1])
        split_state = split[0][2:].split(',')
        state = tuple([int(elm) for elm in split_state])
        sa = (state, action)
    else:
        sa = None
    return sa

#  Compute the domain of each feature representing a state
#  Input: environment name (str), environment
#  Output: features domain dictionary (int-list dict)
def feat_domains(env_name, env):
    if env_name == 'FL':
        feat_domains = {i: [] for i in range(len(list(env.P.keys())[0]))}
        # position
        feat_domains[0] = [i for i in range(env.nRow * env.nCol)]
        # previous position
        # hole position
        previous_pos = []
        hole_pos = []
        for i in range(env.nRow):
            for j in range(env.nCol):
                if bytes(env.desc[i, j]) in b"H":  # hole
                    hole_pos.append(env.to_s(i, j))
                else:  # previous position
                    previous_pos.append(env.to_s(i, j))
        feat_domains[1] = previous_pos
        feat_domains[2] = hole_pos
        # manhattan distance
        feat_domains[3] = [i for i in range(env.nRow + env.nCol - 1)]
        # number of holes
        feat_domains[4] = [env.hole_cpt]
    else:
        feat_domains = None

    return feat_domains

#  Compute several ESX with and without sampling approach, average results and store them into a CSV file
#  Input: CSV file name including state-action pairs to explain (str), environment name (str), explainer instance (ESX),
#  number of samples (int list), CSV file name to store results (str), number of run to perform for each sampling
#  approach (int), CSV file name to store logs (str)
#  Output: None
def compute_explanations(filename, env_name, esx, samples, output_filename, runs=10, esx_csv='trash.csv'):

    file = open(filename, 'r')
    str_sa_list = file.readlines()
    file_lines = []

    # Create the first line
    first_new_line = ['state-action', 'actual score', 'actual runtime']
    for sample in samples:
        first_new_line.append(str(sample)+'s diff score')
        first_new_line.append(str(sample)+'s diff runtime')
    file_lines.append(first_new_line)

    # Compute several ESX's for each state-action couple
    for str_sa in str_sa_list:
        line = []

        # Convert state action couple
        sa = from_str_to_sa(env_name, str_sa[:-1])
        line.append(str(sa))

        # Compute ESX without sample-based approximate approach
        _, scores, runtimes = esx.explain('no_user', sa, ['exh'], 0, esx_csv)
        actual_score, actual_runtime = scores[0], runtimes[0]
        line.append(actual_score)
        line.append(actual_runtime)

        # Compute several ESX's for each number of samples
        for sample in samples:
            sample_scores, sample_runtimes = [], []

            # perform several runs
            for i in range(runs):
                _, scores, runtimes = esx.explain('no_user', sa, ['exh'], sample, esx_csv)
                sample_scores.append(scores[0])
                sample_runtimes.append(runtimes[0])

            # compute avg scores, runtimes and differences
            avg_score = sum(sample_scores) / len(sample_scores)
            avg_runtime = sum(sample_runtimes) / len(sample_runtimes)
            diff_score = avg_score - actual_score
            diff_runtime = avg_runtime - actual_runtime

            # compute and store differences
            line.append(diff_score)
            line.append(diff_runtime)

        #print(new_line)
        file_lines.append(line)

    # Compute the average of score and runtimes over all state-action couple
    last_line = ['Average']
    tmp_file_lines = [line[1:] for line in file_lines[1:]]
    column_file_lines = list(zip(*tmp_file_lines))
    for column in column_file_lines:
        last_line.append(sum(column) / len(column))
    file_lines.append(last_line)

    # Write the result into
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for line in file_lines:
            writer.writerow(line)

    # Remove useless esx file
    os.remove(esx_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-env', '--env_name', default="FL", help="Environment name", type=str, required=True)
    parser.add_argument('-map', '--map_name', default="4x4", help="Map's dimension (nxn)", type=str, required=False)
    parser.add_argument('-policy', '--policy_name', default="4x4", help="Common part of policies name", type=str, required=False)
    parser.add_argument('-k', '--length_k', default=4, help="Length of scenarios", type=int, required=False)
    parser.add_argument('-file', '--sa_filename', default="", help="file containing state-action couples", type=str, required=True)
    parser.add_argument('-csv', '--csv_filename', default="", help="output file", type=str, required=True)
    parser.add_argument('-pred_size', '--pred_size', default='[2,2]', help="predicate size constraints for ESX (expected format: [int,int])", type=str, required=False)
    parser.add_argument('-alpha', '--alpha', default='1.0', help="representative predicate threshold", type=str, required=False)
    parser.add_argument('-samples', '--complement_samples', default='[50,100,200,300]', help="List of samples to test (format: [int,int,int])", type=str, required=False)

    args = parser.parse_args()

    # Get arguments
    PROBLEM = args.env_name
    MAP_NAME = args.map_name
    POLICY_NAME = args.policy_name
    K = args.length_k
    FILENAME = args.sa_filename
    CSV_FILENAME = args.csv_filename
    pred_size = args.pred_size
    ALPHA = float(args.alpha)
    samples = args.complement_samples

    # Convert arguments
        # samples
    split = samples[1:-1].split(',')
    SAMPLES = [int(elm) for elm in split]
        # pred size
    split = pred_size.split(',')
    PRED_SIZE = [int(split[0][1:]), int(split[1][:-1])] # maximal number of terms, literals

    if PROBLEM == 'FL':
        # get directory paths
        agent_Q_dirpath = "Q-tables" + os.sep + "Agent"
        FILENAME = 'FrozenLake' + os.sep + 'Explanation' + os.sep + FILENAME
        CSV_FILENAME = 'FrozenLake' + os.sep + "Explanation" + os.sep + "diff_repr_score" + os.sep + CSV_FILENAME

        # inits
            # env
        env = MyFrozenLake(map_name=MAP_NAME, slip_probas=[0.2, 0.6, 0.2], many_features=True)
            # agent
        agent = Agent(POLICY_NAME, env)
        agent.load(agent_Q_dirpath)
            # ESX instance
        feat_domains = feat_domains(PROBLEM, env)
        functions = [FL_transition, FL_terminal, FL_complement, FL_preprocess, FL_readable_features]
        add_info = {}
        esx = ESX('FL', agent, env, K, feat_domains, PRED_SIZE, ALPHA, functions, add_info)

    else:
        esx = None

    # run experiments
    compute_explanations(FILENAME, PROBLEM, esx, SAMPLES, CSV_FILENAME)
