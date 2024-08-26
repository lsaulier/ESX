import csv
import itertools
import math
import time
from copy import deepcopy
import numpy as np

# In the following code, a tree is a dictionary composed of nodes
# A node is defined by a string, and composed of 6 data
# node (str): - condition (function): condition represented by the node
#             - feature (int): feature id of the condition
#             - value (int): value of the condition
#             - operator (str): condition operator
#             - left (str): name of left child
#             - right (str): name of right child

#  Argmax function
def argmax(array):
    array = list(array)
    return array.index(max(array))

#  Compute the respect probability of a predicate in a set of states
#  Input: predicate (function (list)), state-probability list (state-float list), associated probability (float),
#  features list (int list (list)), negate the predicate or not (bool)
#  Output: weighted probability of a state in S_ respecting d (float)
def predicate_respect(d, S_, p_S_, feats, negation=False):

    p = 0.0
    for state, pr in S_:
        # extract features for predicate check
        if isinstance(d, list):  # case: multiple terms (DNF form)
            feats_values = [[state[i] for i in fts] for fts in feats]
        else:
            feats_values = [state[i] for i in feats]

        # check predicate respect
        if isinstance(d, list):  # case: multiple terms (DNF form)
            respect = not(any([term(*feats_values[i]) for i,term in enumerate(d)])) if negation else any([term(*feats_values[i]) for i,term in enumerate(d)])
            #print('DNF respect: {} with values {}'.format(respect, feats_values))

        else:
            respect = not(d(*feats_values)) if negation else d(*feats_values)

        if respect:
            p += pr

    # weight the result with the set probability
    return p * p_S_

#  Compute phi coefficient
#  Input: predicate (function), state-probability lists (state-float list), associated probabilities (float)
#  Output: phi coefficient (float)
def phi_coef(pred, S, neg_S, p_S, p_neg_S):
    d, feats = pred

    # compute probabilities
    w = predicate_respect(d, S, p_S, feats)
    x = predicate_respect(d, neg_S, p_neg_S, feats)
    y = predicate_respect(d, S, p_S, feats, True)
    z = predicate_respect(d, neg_S, p_neg_S, feats, True)
    #print('w:', w, 'x:', x, 'y:', y, 'z:', z)

    # compute phi coefficient
    cond = w+x == 0 or y+z == 0 or w+y == 0 or x+z == 0
    phi = 0.0 if cond else (w*z - x*y) / math.sqrt((w + x) * (y + z) * (w + y) * (x + z))

    return phi

#  Create a combined lambda function which represents a potential term of the predicate
#  Input: number of literals (int), literals (lambda function list)
#  Output: term (lambda function)
def create_combined_lambda(num_params, operations):
    # Generate initial lambda function that returns the parameters as a tuple
    combined_lambda = lambda *params: params

    # Ensure operations match the number of parameters
    if len(operations) != num_params:
        raise ValueError("Number of operations must match the number of parameters")

    # Iteratively build the combined lambda function
    for i, operation in enumerate(operations):
        previous_lambda = combined_lambda
        combined_lambda = lambda *params, operation=operation, previous_lambda=previous_lambda, index=i: tuple(
            operation(previous_lambda(*params)[index]) if j == index else previous_lambda(*params)[j] for j in
            range(num_params)
        )

    # Final combined lambda to produce a single boolean result
    def final_combined_lambda(*params):
        results = combined_lambda(*params)
        return all(results)  # Example operation: combine all results with AND

    return final_combined_lambda

class ESX:

    def __init__(self, name, agent, env, k, feat_domains, size_constraint, alpha, functions, add_info, context='mono'):
        self.name = name
        self.context = context
        self.agent = agent
        self.env = env
        self.k = k
        self.feat_domains = feat_domains
        self.size_constraint = size_constraint
        self.alpha = alpha
        self.add_info = add_info
        # functions
        transition, terminal, complement, preprocess, human_feats = functions
        self.transition = transition
        self.terminal = terminal
        self.complement = complement
        self.preprocess = preprocess
        # readable features
        self.human_feats = human_feats

    #  Explain an agent's action using ESX
    #  Input: explanation mode (user/no_user/compare) (str), state-action pair, predicate (str), scenario generation
    #  method (str list), number of samples to approximate neg_S (int), csv file for logs (str)
    #  Output: explanations, representativeness scores (float list), ESX runtimes (float list)
    def explain(self, mode, state_action, approaches=[], sample=0, csv_file='logs.csv'):
        ESXs = []
        Scores = []
        Runtimes = []
        # User mode
        if mode == 'user':
            answer = False
            good_answers = ["yes", "y"]
            while not answer:
                question = "Do you want a ESX?"
                esx = input(question)
                if esx in good_answers:
                    k_question = "Which horizon do you want to look at?"
                    self.k = int(input(k_question))
                    # Compute an (approx.) ESX
                    esx, score, runtime = self.esx(state_action, approaches[0], sample, csv_file)
                    ESXs.append(esx)
                    Scores.append(score)
                    Runtimes.append(runtime)

                answer = True
        else:
            # No user mode
            if mode == 'no_user':
                # Compute an (approx.) ESX
                esx, score, runtime = self.esx(state_action, approaches[0], sample, csv_file)
                ESXs.append(esx)
                Scores.append(score)
                Runtimes.append(runtime)

            # Compare mode
            else:
                # Compute an (approx.) ESX
                for approach in approaches:
                    esx, score, runtime = self.esx(state_action, approach, sample, csv_file)
                    ESXs.append(esx)
                    Scores.append(score)
                    Runtimes.append(runtime)

                # Display runtimes
                print('------ Runtimes -------')
                for i, r in enumerate(Runtimes):
                    print('Approach: {} -- Runtime: {}'.format(approaches[i], r))
                print('-----------------------')

        return ESXs, Scores, Runtimes

    #  Compute an ESX
    #  Input: state-action pair, predicate (str), scenario generation method (str), number of samples to approximate
    #  neg_S (int), element to explain (str), csv file for logs (str)
    #  Output: explanation, representativeness score (float), ESX runtime (float)
    def esx(self, state_action, approach, sample, csv_file):
        # Init -----
        esx = []
        env_copy, agent_copy = self.preprocess(self.env, self.agent, 'copies', None, self.add_info) # e.g. deep copies
        start_time = time.perf_counter()

        # ESX -----

        # Generate S (probability,state)
        S = self.first_step(state_action, env_copy)
        S = self.succ(S, self.k - 1, approach, env_copy, agent_copy)
        #print('S: {}'.format(S))

        # Merge similar states in S
        S = self.merge(S)

        # Generate neg_S, p_S, p_neg_S
        neg_S, p_S, p_neg_S = self.complement(S, sample, env_copy)
        #print('neg_S: {}'.format(neg_S))
        #print('p_S: {}'.format(p_S))
        #print('p_neg_S: {}'.format(p_neg_S))

        # Build a tree which contains a set of representative terms
        T = self.tree(S, neg_S, p_S, p_neg_S)
        print('Tree: {}'.format(T))

        # Extract a representative predicate
        d_func, d_nodes, feats, score = self.extract(T, S, neg_S, p_S, p_neg_S)

        # Get predicate percentage of cover of S and neg_S
        self.coverage(d_func, feats, S, neg_S)

        esx.append([d_func, d_nodes])
        self.save(T, d_nodes, score, csv_file)

        final_time = time.perf_counter() - start_time
        print("-------------------------------------------")
        print("Explanation achieved in: {} second(s)".format(final_time))
        print("-------------------------------------------")

        return esx, score, final_time

    #  Generate the scenarios (in an exhaustive or approximate way)
    #  Input: set of states obtained after 1 step starting by doing an action from s (state-probability list),
    #  scenarios length (int), scenario generation method (str), environment (Environment), agent(s) list (Agent (list)),
    #  Output: set of states obtained after k+1 steps starting by doing an action from s (state-probability list)
    def succ(self, S, k, approach, env, agent):
        S_tmp = []

        if approach != 'exh':
            det_transition = int(approach.split('_')[1])
            approx_mode = approach.split('_')[0]
        else:
            det_transition = 0
            approx_mode = 'none'

        # Limit the number of last det. transition to the depth of scenarios
        if approx_mode == 'last' and det_transition > k:
            det_transition = k

        # Determine the number of exhaustive step(s)
        exh_steps = k if approx_mode == 'none' else k - det_transition if approx_mode == 'last' else 0

        # Generate scenarios
        for _ in range(k):

            # print('len(S): {}'.format(len(S)))
            for s in S:
                # Extract state, proba
                state, proba = s

                if not self.is_terminal(state, env):
                    action = self.predict(agent, state)
                    # print('from state: {} -- action: {}'.format([s[1] for s in state], action))
                    for p, new_s in self.transition(state, action, env, approx_mode, exh_steps, det_transition, self.add_info):
                        S_tmp.append((new_s, proba * p))

                else:
                    # Add the terminal state
                    S_tmp.append((state, proba))

            S = S_tmp
            S_tmp = []

            exh_steps -= 1 if exh_steps else 0

        return S

    #  Get first set of states for the succ function (scenarios generation)
    #  Input: state-action pair, environment (Environment)
    #  Output: set of states by doing pi(s) from s (state-probability list)
    def first_step(self, state_action, env):
        S = []
        s, a = state_action
        # print('s: {}'.format(s))
        # print('a: {}'.format(a))

        for p, new_s in self.transition(s, a, env, 'none', add_info=self.add_info):
            S.append((new_s, p))

        return S

    #  Check whether the state is terminal or not
    #  Input: state (state), environment (Environment)
    #  Output: (bool)
    def is_terminal(self, state, env):
        return self.terminal(state, env, self.add_info)

    #  Predict the agent(s) action from a specific state using its (their) policy
    #  Input: agent or agents list (Agent (list)) and state or states list (state (list))
    #  Output: action or actions list (int (list))
    def predict(self, agent, state):
        action = None
        # mono agent setting
        if self.context == 'mono':
            if 'net' in self.add_info:
                if self.name == 'C4':
                    action = agent.predict(state, net=self.add_info['net'])
                elif self.name == 'DO':
                    action, _ = agent.predict(np.array(state[0]), deterministic=True)
            else:
                if self.name in ['FL','BJ']:
                    action = agent.predict(state)
                else:
                    action = agent[state]

        # multi-agent setting
        else:
            if self.name == 'DC':
                action = [ag.predict(self.add_info['net'], obs=state[idx]) for idx, ag in enumerate(agent)]
            else:
                action = None

        return action

    #  Merge probabilities of similar states
    #  Input: state-probability list (state-float list)
    #  Output: state-probability list (state-float list)
    def merge(self, S):
        # store in a dict state and probabilities
        tmp_dict = {}
        for s,pr in S:
            if tmp_dict.get(s):
                tmp_dict[s] += pr
            else:
                tmp_dict[s] = pr

        # convert dict into list
        S = list(tmp_dict.items())

        #print(S)
        return S

    #  Build the predicate tree
    #  Input: state-probability lists (state-float list), set probabilities (float)
    #  Output: tree (str-dict dictionary)
    def tree(self, S, neg_S, p_S, p_neg_S):
        _, literal_max = self.size_constraint
        T = {}
        ct_node_id = 0

        # Select root literal and update the Tree
        l = self.select(T, [], S, neg_S, p_S, p_neg_S)
        T, ct_node_id = self.update(T, [], l, ct_node_id)
        #print()
        #print('tree after init: {}'.format(T))

        for i in range(literal_max - 1): # Tree depth is imposed with literal_max

            branches = self.get_branches(T)
            for b in branches:
                print('Focus: {}'.format(b))

                # Get representative literals for the current branch
                l_left = self.select(T, b, S, neg_S, p_S, p_neg_S)
                print('---> for left child')
                print()

                l_right = self.select(T, b, S, neg_S, p_S, p_neg_S, True) # use negation of the last branch condition
                print('---> for right child')
                print()

                print('selected literals: left {}, right {}'.format(l_left, l_right))
                print()

                # Update it
                T, ct_node_id = self.update(T, b, l_left, ct_node_id)
                T, ct_node_id = self.update(T, b, l_right, ct_node_id, True)
                #print('tree after update: {}'.format(T))

        #print('Number of nodes: {}'.format(len(T)))
        #print('Tree: {}'.format(T))
        return T

    #  Select the literal which maximises the current branch representativeness
    #  Input: branch (str list), state-probability lists (state-float list), set probabilities (float),
    #  negative condition or not of the last branch condition (bool)
    #  Output: literal feature (int), condition value (int) and operator (str), condition (lambda function)
    def select(self, T, b, S, neg_S, p_S, p_neg_S, negation=False):
        feature = -1
        condition = True
        score = -2
        value = None
        operator = '?'
        feats_domains = deepcopy(self.feat_domains)
        S_Pr = [S, neg_S, p_S, p_neg_S]

        # Remove features that already are in the branch
        for node in b:
            del feats_domains[T[node]['feature']]

        # Test each condition for each feature (keep the most representative one)

        # get branch conditions (reminder: predicate = function - list of features)
        b_conditions, b_feats = self.get_conditions(T, b, negation)
        #print('actual b conditions and associated feats: {} -- {}'.format(b_conditions, b_feats))

        # feature loop
        for feat, values in feats_domains.items():
            print('tested feature: {} - values: {}'.format(feat,values))
            # condition loop
            for v in values:
                # test condition x >= v
                feature, condition, score, value, operator, b_conditions, b_feats = self.test_condition(feat, v,
                                                b_feats, b_conditions, feature, condition, score, value, operator, S_Pr)
                #print(feature, operator, value, score)

                # test condition x < v
                feature, condition, score, value, operator, b_conditions, b_feats = self.test_condition(feat, v,
                                b_feats, b_conditions, feature, condition, score, value, operator, S_Pr,True)
                #print(feature, operator, value, score)

        print('feature: {} - value {} - operator {} - score: {}'.format(feature, value, operator, score))
        return feature, value, operator, condition

    #  Test a condition to add in a branch
    #  Input: condition feature / value (int), current features/conditions in the branch (int/function list),
    #  current best feature/condition/score/value/operator (int/function/float/int/str), list of state-probability sets
    #  and set probabilities, negative condition or not (bool)
    #  Output: current best feature/condition/score/value/operator (int/function/float/int/str), current
    #  features/conditions in the branch (int/function list)
    def test_condition(self, feat, v, b_feats, b_conditions, feature, condition, score, value, operator, S_Pr, negation=False):
        S, neg_S, p_S, p_neg_S = S_Pr

        # create tmp condition
        if not negation:
            test_cond = lambda x: x >= v
            #print('f' + str(feat),'>=', v)
        else:
            test_cond = lambda x: x < v
            #print('f' + str(feat),'<', v)

        # merge it with branch predicate
        b_conditions.append(test_cond)
        b_feats.append(feat)

        # create tmp rule
        tmp_rule = create_combined_lambda(len(b_feats), b_conditions)

        # use phi coefficient to compare it (test also the negation of condition?)
        phi = phi_coef((tmp_rule, b_feats), S, neg_S, p_S, p_neg_S)

        # keep the best one
        if phi > score:
            feature = feat
            condition = test_cond
            score = phi
            value = v
            if not negation:
                operator = '>='
                #print('new best rule: f{} >= {}, score {}'.format(feature, v, score))
            else:
                operator = '<'
                #print('new best rule: f{} < {}, score {}'.format(feature, v, score))

        # remove tested condition and associated feature
        b_conditions.remove(test_cond)
        b_feats.remove(feat)

        return feature, condition, score, value, operator, b_conditions, b_feats

    #  Extract conditions and features from branch nodes
    #  Input: tree (str-dict dictionary), branch (str list), negate the last branch condition or not (bool)
    #  Output: condition list (function list), features list (int list)
    def get_conditions(self, T, b, negation):
        conditions = []
        feats = []

        # Get feature and condition list
        previous_node = None
        for node in b:
            # if node is right child of previous node, negate the previous node
            if previous_node is not None and T[previous_node]['right'] == node:
                conditions[-1] = self.negate(T, previous_node)

            conditions.append(T[node]['condition'])
            feats.append(T[node]['feature'])
            previous_node = node

        # Negate last condition
        if negation:
            conditions[-1] = self.negate(T, b[-1])

        return conditions, feats

    #  Negate a literal
    #  Input: tree (str-dict dictionary), node (str)
    #  Output: literal (lambda function)
    def negate(self, T, node):
        v = T[node]['value']
        # print('negation: v {}, operator {}'.format(v, T[b[-1]]['operator']))

        if T[node]['operator'] == '>=':
            cond = lambda x: x < v
        else:
            cond = lambda x: x >= v

        return cond

    #  Update a branch of the tree by adding a new node
    #  Input: tree (str-dict dictionary), branch (str list), node info (int-function), counter for node name (int),
    #  negative condition or not (bool)
    #  Output: tree (str-dict dictionary), counter for node name (int)
    def update(self, T, b, l, ct, negation=False):
        feat, v, op, cond = l
        ct += 1
        node_id = 'node_' + str(ct)
        #print('update: new node {}'.format(node_id))

        # Create new node
        T[node_id] = {'feature': feat, 'value': v, 'operator': op, 'condition': cond, 'left': None, 'right': None}

        # Add as left or right child to parent node (if it's not the tree root)
        if b:
            if not negation:  # case: left child
                T[b[-1]]['left'] = node_id
            else:  # case: right child
                T[b[-1]]['right'] = node_id

        return T, ct

    #  Extract (partial) branches from the tree
    #  Input: tree (str-dict dictionary), extract partial branches or not (bool)
    #  Output: branch list (str list list)
    def get_branches(self, T, partial=False):
        branches = []

        if partial: # a partial term which only contains the root node
            branches.append(['node_1'])

        tmp_branches = [['node_1']]

        while not all([T[b[-1]]['left'] is None for b in tmp_branches]): # while the end of the tree is not reached

            # Extend current branches
            new_b = []
            for b in tmp_branches:
                left_node, right_node = T[b[-1]]['left'], T[b[-1]]['right']
                #print('left_node: {}, right_node: {}'.format(left_node, right_node))

                new_b.append(b + [left_node])
                new_b.append(b + [right_node])

            if partial: # add partial branches
                branches.extend(new_b)

            # Replace branches
            tmp_branches = new_b
            #print(tmp_branches)

        if not partial: # add entire branches
            branches = tmp_branches

        return branches

    #  Extract a representative predicate from the tree, and the associated ids of features
    #  Input: tree (str-dict dictionary), state-probability lists (state-float list), set probabilities (flot)
    #  Output: predicate (lambda function), string predicate (str), id of features (int list), representativeness score
    #  (float)
    def extract(self, T, S, neg_S, p_S, p_neg_S):
        pred_function = None
        pred_nodes = None
        feats = None
        terms_max, _ = self.size_constraint
        alpha = self.alpha
        score = -2

        # extract each (partial) branch which contains root node
        partial_branches = self.get_branches(T, True)
        print('partial_branches: {}'.format(partial_branches))

        # get all combinations of maximum terms_max branches
        for i in range(1, terms_max + 1):
            for dnf in itertools.combinations(partial_branches, i):

                tmp_formula = []
                tmp_feats = []
                for term in dnf:
                    # get branch conditions (reminder: predicate = function - list of features)
                    b_conditions, b_feats = self.get_conditions(T, term, False)

                    # create rule
                    tmp_formula.append(create_combined_lambda(len(b_feats), b_conditions))
                    tmp_feats.append(b_feats)

                # compute phi coefficient
                phi = phi_coef((tmp_formula, tmp_feats), S, neg_S, p_S, p_neg_S)
                print('predicate: {} score {}'.format(dnf,phi))

                if phi > score:
                    score = phi
                    pred_function = tmp_formula
                    pred_nodes = dnf
                    feats = tmp_feats
                    if score > alpha: # stop search when a sufficient representative predicate is found
                        break
            else:
                continue
            break

        return pred_function, pred_nodes, feats, score

    #  Compute the probability respect of a predicate for two sets of states
    #  Input: predicate (lambda function), features list (int list (list)), state-probability lists (state-float list)
    #  Output: None
    def coverage(self, d, feats, S, neg_S):
        # compute d coverage of S
        cov_S = predicate_respect(d, S, 1.0, feats)

        # compute d coverage of neg_S
        cov_neg_S = predicate_respect(d, neg_S, 1.0, feats)

        print('Respect probability in S: {}'.format(cov_S))
        print('Respect probability in neg_S: {}'.format(cov_neg_S))
        return

    #  Save the tree and extracted predicate into a CSV file
    #  Input: tree (str-dict dictionary), predicate (str list (list)), representativeness score (float), csv file (str)
    #  Output: None
    def save(self, T, d, score, csv_file):
        with open(csv_file, 'w') as file:
            writer = csv.writer(file)

            # head of the file
            writer.writerow(['Node name', 'Expression', 'Left child', 'Right child'])

            # write the tree level by level
            levels = 1 + math.floor(math.log2(len(T)))
            for i in range(levels):
                writer.writerow(['Level: ' + str(i+1)])
                level_node_ct = 2**i
                for j in range(level_node_ct):
                    node_name = 'node_' + str(level_node_ct + j)
                    node_expression = 'f' + str(T[node_name]['feature']) + ' ' + T[node_name]['operator'] + ' ' + str(T[node_name]['value'])
                    writer.writerow([node_name, node_expression, T[node_name]['left'], T[node_name]['right']])

            # output a readable predicate in the console
            readable_d = self.render(T, d, score)
            writer.writerow(['Predicate', readable_d, 'Representativeness', score])

        return

    #  Display the representative predicate in a human-readable fashion
    #  Input: tree (str-dict dictionary), predicate (str list (list)), representativeness score (float)
    #  Output: readable predicate (str)
    def render(self, T, d, score):
        feats = self.human_feats
        predicate = ''

        for i, term in enumerate(d):
            if i: predicate += ' OR '

            for j, node in enumerate(term):
                if not j : predicate += '('
                else: predicate += ' AND '

                feature = feats[T[node]['feature']]
                value = T[node]['value']

                # negate operator if the next node is right child
                if j + 1 < len(term) and term[j + 1] == T[node]['right']:
                    operator = '<' if T[node]['operator'] == '>=' else '>='
                else:
                    operator = T[node]['operator']

                predicate += str(feature) + ' ' + operator + ' ' + str(value)

            predicate = predicate + ')'

        print('Predicate: {}'.format(predicate))
        print('Score: {}'.format(score))
        return predicate
