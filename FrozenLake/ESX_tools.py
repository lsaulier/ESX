import random
from copy import deepcopy

# Human-readable features
readable_features = ['P', 'PP', 'HP', 'PD', 'HN']

#  Get from a state-action couple, the entire/part of the transitions available, i.e. the new states associated with
#  their probabilities
#  Input: agent's state (int (list)), action (int), environment (MyFrozenLake), importance score method (str), number of
#  exhaustive/deterministic steps (int), additional information (dictionary), importance type (str)
#  Output: list of transition-probability couples (couple list)
def transition(s, a, env, approx_mode, exh_steps=0, det_tr=0, add_info=None, imp_type=None):
    transitions = [(t[0], t[1]) for t in deepcopy(env.P[s][a])]

    # Look all possible transitions from s
    if approx_mode == 'none' or exh_steps:
        return transitions

    else:
        # Look for the most probable transition
        if approx_mode == 'last':
            # Specific case: equiprobable transitions
            return extract_transitions(1, transitions, approx_mode)
        # Select the 'det' most probable transition(s)
        else:
            return extract_transitions(det_tr, transitions, approx_mode)

#  Check whether the state is terminal or not
#  Input: state (int), environment (MyFrozenLake), additional information (dict)
#  Output: (bool)
def terminal(s, env, add_info):
    state = s if not env.many_features else s[0]
    row, col = state // env.nCol, state % env.nCol

    return bytes(env.desc[row, col]) in b"GH"

#  Get (partial) complement of S and get probability of S and neg_S
#  Input: states list (int list or tuple int list), number of samples (int), environment (MyFrozenLake)
#  Output: states list (int list or tuple int list)
def complement(S, sample, env):
    neg_S = []
    states_S = [s for s,pr in S]

    # exhaustive search of states
    if not sample:
        for key in env.P:
            if key not in states_S:
                neg_S.append(key)

        # add a similar probability to each state
        for i in range(len(neg_S)):
            neg_S[i] = (neg_S[i], 1 / len(neg_S))

    # uniform sampling states
    else:
        for i in range(sample):
            state = generate(env)
            while state in states_S or state in neg_S:
                state = generate(env)
            neg_S.append((state, 1/sample))

    # get probabilities of S and neg_S
    nb_states = len(env.P)
    p_S = len(S) / nb_states
    p_neg_S = len(neg_S) / nb_states

    #print('p_S: {}'.format(p_S))
    #print('p_neg_S: {}'.format(p_neg_S))
    #print('########################')

    if sample: # evenly split probability in case states are not represented in S and neg_S
        p = (1.0 - p_S - p_neg_S) / 2
        p_S += p
        p_neg_S += p

    return  neg_S, p_S, p_neg_S

#  Randomly produce a state
#  Input: environment (MyFrozenLake)
#  Output: state (int or int tuple)
def generate(env):
    return random.choice(list(env.P.keys()))

#  Compute the argmax of an array
#  Input: an array (list)
#  Output: index of the maximum value (int)
def argmax(array):
    array = list(array)
    return array.index(max(array))

#  Extract n most probable transitions
#  Input: number of transition to extract (int), transitions (tuple list), importance score method (str)
#  Output: most probable transition(s) (tuple list)
def extract_transitions(n, transitions, approx_mode):
    most_probable = []

    while n != len(most_probable):
        probas = [t[0] for t in transitions]
        max_pr, idx_max_pr = max(probas), argmax(probas)
        tmp_cpt = probas.count(max_pr)
        # Only one transition is the current most probable one
        if tmp_cpt == 1:
            temp_t = list(transitions[idx_max_pr])
            most_probable.append(temp_t)
            transitions.remove(transitions[idx_max_pr])

        else:
            # There are more transitions than wanted (random pick)
            if tmp_cpt > n - len(most_probable):
                random_tr = random.choice([t for t in transitions if t[0] == max_pr])
                temp_random_tr = list(random_tr)
                most_probable.append(temp_random_tr)
                transitions.remove(random_tr)

            else:
                tmp_list = []
                for t in transitions:
                    if t[0] == max_pr:
                        temp_t = list(t)
                        most_probable.append(temp_t)
                        tmp_list.append(t)
                for t in tmp_list:
                    transitions.remove(t)

    # Probability distribution
    sum_pr = sum([p for p, s in most_probable])
    if sum_pr != 1.0:
        delta = 1.0 - sum_pr
        add_p = delta / len(most_probable)
        for elm in most_probable:
            elm[0] += add_p
    return most_probable

#  Multiple-tasks function to update some data during the HXP process
#  Input: environment (MyFrozenLake), agent (Agent), location of the modification in the ESX process (str),
#  state-action list (int (list)-int list), additional information (dictionary)
#  Output: variable
def preprocess(env, agent, location, s_a_list=None, add_info=None):
    return env, agent

#  Render the most important action(s) / transition(s)
#  Input: state-action to display (int (list)-int  list), environment (MyFrozenLake), agent (Agent),
#  importance type (str), runtime (float), additional information (dictionary)
#  Output: None
def render(esx, env, agent, imp_type, runtime, add_info):
    # Render
    env_copy = deepcopy(env)
    for s_a_list, i in esx:
        print("Timestep {}".format(i))
        env_copy.setObs(s_a_list[0])
        env_copy.render()
        print("    ({})".format(["Left", "Down", "Right", "Up"][s_a_list[1]]))
        if imp_type == 'transition':
            env_copy.setObs(s_a_list[2])
            env_copy.render()
    # Runtime
    print("-------------------------------------------")
    print("Explanation achieved in: {} second(s)".format(runtime))
    print("-------------------------------------------")
    return
