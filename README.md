The following repo contains the code for the ESX method presented in the thesis 'Formal Reasoning for Reinforcement Learning' (TODO change name).
The implementation is composed of *1* folder. The code was done under the 3.8 Python version.  Before testing our implementation, it's *necessary* to install packages of requirements.txt using the 
following pip command: 

```bash
pip install -r requirements.txt
```

Then, before running test or training file, the user must be in the *Frozen Lake* directory:
```bash
cd 'FrozenLake'
```

Find below the main commands to use:
```bash
#####  Frozen Lake  ##### 
# Training of an Agent for the 8x8 map with 10,000 episodes. The name of the trained policy is '8x8_test' (not required command). An agent's state is composed of 5 features.
python3 train.py -policy '8x8_test' -features
# Test the default policy trained in a 8x8 map. By default, the user can ask at each timestep, an ESX. An agent's state is composed of 5 features.
python3 test.py -map 8x8 -policy 'features_8x8_proba262'
# Compute an ESX from a specific state-action pair ([(39,38,46,11,10),1]) using the default policy of the 8x8 map. The predicate is composed of at most 2 terms, each of them composed of at most 3 literals. 
# The horizon for the scenario generation is 4. The predicate search stop when the representativeness score of the current predicate is at least 0.9.  
# Generated tree and final predicate are stored in test.csv. An agent's state is composed of 5 features.
python3 test.py -map 8x8 -policy 'features_8x8_proba262' -csv test.csv -pred_size [2,3] -alpha 0.9 -sa '[(39,38,46,11,10),1]' -k 4
```

# Code Structure #


## Frozen Lake (FL) ##

### File Description ###

The Frozen Lake folder is organised as follows:

* **train.py**: parameterized python file which calls training function for Agent instance, and store learnt Q-table into text file.


* **test.py**: parameterized python file which loads learnt Q-table and tests it. This file can be use in two ways:
    * A classic sequence loop (named *user mode*) which starts in the initial state of the chosen map. The agent's policy is used and must be provided by the user if the map is not '4x4'. 
      At each time-step, the user can ask for an ESX.
    * A specific computation of an ESX from a given state-action pair. In this case, the user must provide at least the *-sa* parameter.


* **agent.py**: contains the *Agent* class for the RL agent.


* **env.py**: contains the *MyFrozenLake* class: a variant of the Frozen Lake environment (cf. https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py). Two state representations are available: the first one is simply composed of the agent's position while the second one is composed of 5 features: agent position (P), agent's previous position (PP), position of one of the two holes closest to the agent (HP), the Manhattan distance between the agent's initial position and his current position (PD), and the total number of holes on the map (HN). 


* **ESX_tools.py**: set of specific functions used for the ESX computation.


* **Q-tables folder**: contains all learnt Q-tables. Each file name starts by *'Q_'*.


* **Explanation folder**: contains text files storing lists of state-action pairs and different folders including CSV files. Each file in the *diff_repr_score* folder contains results of ESX computation based on a list of state-action pairs. Folders entitled by a map name (e.g. *8x8*) include logs of ESX computation (i.e. generated tree and final predicate).


By default, running **train.py** starts a training of Agent of 10,000 episodes, on the 4x4 map and **test.py**
runs a classic testing loop on the 4x4 map. To test ESX for other agents and maps, the user must set the parameters *-map* and *-policy*. 

### Examples ###

The followings bash commands are examples of use of **train.py** and **test.py** files.

**Train:**
```bash
# Training of an Agent for the 4x4 map with 10,000 episodes. The name of the trained policy is '4x4_test' (not required command)
python3 train.py -policy '4x4_test'
# Training of an Agent on 10x10 map with 500,000 episodes and save Q-table in text file with a name finishing by "10x10_test"
python3 train.py -map "10x10" -policy "10x10_test" -ep 500000
```
**Test:**
```bash
#####  Test in user mode a policy  ##### 

# Test the default policy trained in a 6x6 map. The user can ask at each timestep, an ESX in an exhaustive way. An agent's state is composed of 5 features.
# By default, the maximal number of terms and literals in the sought predicate are respectively 2 and 2. Moreover, the predicate search stop when the representativeness score of the current predicate is at least 0.8. 
python3 test.py  -map 6x6 -policy 'features_6x6_proba262'
# Test the default policy trained in a 8x8 map. The user can ask at each timestep, an ESX in an approximate way (i.e. the complement of S is composed of 100 randomly drawn states). An agent's state is composed of 5 features.
# The maximal number of terms and literals in the sought predicate are respectively 3 and 2. Moreover, the predicate search stop when the representativeness score of the current predicate is at least 0.5. 
python3 test.py -map 8x8 -policy 'features_8x8_proba262' -pred_size '[3,2]' -samples 100 -alpha 0.5

#####  Test ESX from a specific history  ##### 

# Compute an ESX from a specific state-action pair ([(3,4,14,3,5),2]) using the default policy of the 6x6 map. The predicate is composed of at most 2 terms, each of them composed of at most 2 literals. 
# The horizon for the scenario generation is 2. The predicate search stop when the representativeness score of the current predicate is at least 0.8.  
# Generated tree and final predicate are stored in test.csv. An agent's state is composed of 5 features.
python3 test.py -map 6x6 -policy 'features_6x6_proba262' -csv test.csv -sa '[(3,4,14,3,5),2]' -k 2
# Compute an ESX from a specific state-action pair ([(39,38,46,11,10),1]) using the default policy of the 8x8 map. The predicate is composed of at most 2 terms, each of them composed of at most 3 literals. 
# The horizon for the scenario generation is 4. The predicate search stop when the representativeness score of the current predicate is at least 0.9.  
# Generated tree and final predicate are stored in test.csv. An agent's state is composed of 5 features.
python3 test.py -map 8x8 -policy 'features_8x8_proba262' -csv test.csv -pred_size [2,3] -alpha 0.9 -sa '[(39,38,46,11,10),1]' -k 4
```

## Additional files ##

The following files are located at the root of the project and are used for the computation of ESX's:

* **ESX.py**: given an RL problem, this file allows to perform ESX for a given state-action pair. 


* **diff_repr_score.py**: computes ESX for a set of state-action pairs and stores the results in a csv file. For each pair, the exhaustive approach and the different sampling approaches are calculated. For each sampling approach, several runs are performed and the average result (including the representativeness score and the runtime) is stored. 
                     Find below an example of this file use:
```bash
# In the 8x8 Frozen Lake map, compute ESX in a exhaustive and approximate fashion for each state-action pair in FL_8x8_30sa.txt. The features_8x8_proba262 policy is used to compute ESX's. 
# Results are stored in the test.csv file. By default for each number of samples of the following list: [50,100,200,300], 10 runs of approximate ESX are performed, and the average result is stored. 
# The maximal number of terms and literals in the sought predicate are respectively 2 and 2 by default.
python3 diff_repr_score.py -env FL -map 8x8 -policy features_8x8_proba262 -file FL_8x8_30sa.txt -csv test.csv
```
