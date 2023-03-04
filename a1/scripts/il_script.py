import numpy as np
from sklearn.linear_model import LogisticRegression
from MDP import build_mazeMDP
from DP import DynamicProgramming
np.set_printoptions(threshold=np.inf)

# optimal_pi = np.array([1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 0])
def main():
    mdp = build_mazeMDP()
    dp = DynamicProgramming(mdp)
    expert_pi, _, _ = dp.policyIteration(initial_pi, True)
    dataset = collectData(expert_pi, dp, 75)
    trainModel(dataset[0], dataset[1])

def collectData(expert_policy, dp, sample_size=500):
    """
    Output:
        states: An array of the form [s_1, s_2, ..., s_n], where s_i is a one-hot encoding of the state at timestep i,
                    and n=sample_size
        actions: An array of the form [a_1, a_2, ..., a_n], where a_i is the action taken at timestep i,
                    and n=sample_size

    Hint: You may need to make a deep copy of intermediate arrays (e.g. using np.copy())
    Hint: How do you get an action from your policy? How do you transition
    from one state to another?
    """
    #initialize dataset
    states = []
    actions = []
    
    #initialize state
    s = 0
    # Main data collection loop
    for i in range(sample_size):
        state = np.zeros(17)

        #print("j:   ", j, "s:   ", s)
        action = expert_policy[s]
        #print("action:  ", action)
        prob = dp.P[action][s]
        #print("prob:    ", prob)
        prob_idx = np.nonzero(prob)[0]
        #print("prob_idx:    ", prob_idx)
        p0 = prob[prob_idx]
        states.append(state)
        actions.append(action)
        state[s] = 1
        s = np.random.choice(prob_idx, 1, p=p0)[0]
        # Reset state to 0 if we reach the terminal state
        if s == dp.nStates-1:
            s = 0

    return np.array(states), np.array(actions)

def trainModel(states, actions):
    """
    Uses the dataset to train a policy pi using behavior cloning, using
    scikit-learn's logistic regression module as our policy class.

    Input:
        states: An array of the form [s_1, s_2, ..., s_n], where s_i is a one-hot encoding of the state at timestep i
                Note: n>=1
        actions: An array of the form [a_1, a_2, ..., a_n], where a_i is the action taken at timestep i
                Note: n>=1
    Output:
        pi: the learned policy (array of |S| integers)

    Hint: think about how to convert the output of the logistic regression model
          into a policy with the above specification.
    """
    # TODO: Replace the placeholders with the actual definitions of X and y here,
    # where X and y are the training inputs and outputs for the logistic regression model.
    X = states
    y = actions
    print(X)
    print(y)
    # Learn policy using logistic regression
    clf = LogisticRegression(random_state=0).fit(X, y)
    # Convert policy to vector form
    # Note that in collectData, we collected a dataset (s,a), so when we
    # train using logistic regression, we'll get a model phi that maps a one-hot vector s to
    # an integer a. However, our policy evaluation code requires a vector.
    pi = np.zeros(len(states[0]))
    # TODO: fill in the learned policy pi here using the logistic regression model.
    for s in range(len(states[0])):
        one_hot_s = np.zeros(states[0].shape[0])
        one_hot_s[s] = 1
        one_hot_s = one_hot_s.reshape((1, 17))
        pi[s] = clf.predict(one_hot_s)
    pi = pi.astype(int)
    print(pi.dtype)
    return pi


initial_pi = np.array([2, 3, 3, 2, 1, 1, 0, 3, 1, 2, 3, 0, 0, 0, 1, 2, 1])
initial_V_pi = np.array([-58.584607026434036, -240.99917438690397, -293.4241896968443, -334.4514382946084, 
                         -94.57320482379257, -167.45960248637064, -311.9075404604963, -181.06114530240242, 
                         -92.37900926495321, -177.11857440157402, -136.83415304316443, -154.38281179345748, 
                         -87.29656423805085, -120.83520169624771, 26.18175073245016, 200.0, 0.0])


initial_Q_pi = np.array([[-78.35211291745406, -101.02492952978994, -58.584607026434036, -173.5057844635301], 
                         [-200.3506674213921, -154.02073712405607, -93.05023730454552, -240.99917438690395], 
                         [-263.543072221016, -275.18758320211685, -234.5492634349905, -293.4241896968443], 
                         [-375.4676159044495, -278.8317313193596, -334.45143829460835, -360.2986049111998], 
                         [-73.28273141352547, -94.57320482379255, -80.9612072383266, -126.87963776575079], 
                         [-207.7043804771285, -167.4596024863706, -117.02701512543383, -253.94764657655722], 
                         [-311.90754046049625, -213.25581736867795, -243.5844258363147, -252.1533978104147], 
                         [-278.2551787035946, -164.8119440078695, -263.49437425200153, -181.06114530240242], 
                         [-96.96329283397048, -92.37900926495321, -83.75119466026939, -137.1371206963405], 
                         [-216.44332647800937, -187.06995398023196, -177.118574401574, -205.12531498184705], 
                         [-242.25443762644193, -29.258184174885653, -151.15768348627785, -136.83415304316443], 
                         [-154.38281179345748, 85.68570974705604, -84.64877103301792, -95.70442604570252], 
                         [-87.29656423805083, -84.09462387110234, -80.25303789287759, -101.38237949154161], 
                         [-120.83520169624771, -85.37667689189215, -96.22059524317795, -24.72925681176233], 
                         [-76.51826864618704, 26.181750732450162, -92.06425138058249, 110.06192568805356], 
                         [200.0, 200.0, 200.0, 200.0], 
                         [0.0, 0.0, 0.0, 0.0]])

P = np.zeros([4, 17, 17])
a = 0.7;  # intended move
b = 0.15;  # lateral move
P[0, 0, 0] = a + b
P[0, 0, 1] = b

P[0, 1, 0] = b
P[0, 1, 1] = a
P[0, 1, 2] = b

P[0, 2, 1] = b
P[0, 2, 2] = a
P[0, 2, 3] = b

P[0, 3, 2] = b
P[0, 3, 3] = a + b

P[0, 4, 4] = b
P[0, 4, 0] = a
P[0, 4, 5] = b

P[0, 5, 4] = b
P[0, 5, 1] = a
P[0, 5, 6] = b

P[0, 6, 5] = b
P[0, 6, 2] = a
P[0, 6, 7] = b

P[0, 7, 6] = b
P[0, 7, 3] = a
P[0, 7, 7] = b

P[0, 8, 8] = b
P[0, 8, 4] = a
P[0, 8, 9] = b

P[0, 9, 8] = b
P[0, 9, 5] = a
P[0, 9, 10] = b

P[0, 10, 9] = b
P[0, 10, 6] = a
P[0, 10, 11] = b

P[0, 11, 10] = b
P[0, 11, 7] = a
P[0, 11, 11] = b

P[0, 12, 12] = b
P[0, 12, 8] = a
P[0, 12, 13] = b

P[0, 13, 12] = b
P[0, 13, 9] = a
P[0, 13, 14] = b

P[0, 14, 13] = b
P[0, 14, 10] = a
P[0, 14, 15] = b

P[0, 15, 16] = 1
P[0, 16, 16] = 1

# down (a = 1)

P[1, 0, 0] = b
P[1, 0, 4] = a
P[1, 0, 1] = b

P[1, 1, 0] = b
P[1, 1, 5] = a
P[1, 1, 2] = b

P[1, 2, 1] = b
P[1, 2, 6] = a
P[1, 2, 3] = b

P[1, 3, 2] = b
P[1, 3, 7] = a
P[1, 3, 3] = b

P[1, 4, 4] = b
P[1, 4, 8] = a
P[1, 4, 5] = b

P[1, 5, 4] = b
P[1, 5, 9] = a
P[1, 5, 6] = b

P[1, 6, 5] = b
P[1, 6, 10] = a
P[1, 6, 7] = b

P[1, 7, 6] = b
P[1, 7, 11] = a
P[1, 7, 7] = b

P[1, 8, 8] = b
P[1, 8, 12] = a
P[1, 8, 9] = b

P[1, 9, 8] = b
P[1, 9, 13] = a
P[1, 9, 10] = b

P[1, 10, 9] = b
P[1, 10, 14] = a
P[1, 10, 11] = b

P[1, 11, 10] = b
P[1, 11, 15] = a
P[1, 11, 11] = b

P[1, 12, 12] = a + b
P[1, 12, 13] = b

P[1, 13, 12] = b
P[1, 13, 13] = a
P[1, 13, 14] = b

P[1, 14, 13] = b
P[1, 14, 14] = a
P[1, 14, 15] = b

P[1, 15, 16] = 1
P[1, 16, 16] = 1

# left (a = 2)

P[2, 0, 0] = a + b
P[2, 0, 4] = b

P[2, 1, 1] = b
P[2, 1, 0] = a
P[2, 1, 5] = b

P[2, 2, 2] = b
P[2, 2, 1] = a
P[2, 2, 6] = b

P[2, 3, 3] = b
P[2, 3, 2] = a
P[2, 3, 7] = b

P[2, 4, 0] = b
P[2, 4, 4] = a
P[2, 4, 8] = b

P[2, 5, 1] = b
P[2, 5, 4] = a
P[2, 5, 9] = b

P[2, 6, 2] = b
P[2, 6, 5] = a
P[2, 6, 10] = b

P[2, 7, 3] = b
P[2, 7, 6] = a
P[2, 7, 11] = b

P[2, 8, 4] = b
P[2, 8, 8] = a
P[2, 8, 12] = b

P[2, 9, 5] = b
P[2, 9, 8] = a
P[2, 9, 13] = b

P[2, 10, 6] = b
P[2, 10, 9] = a
P[2, 10, 14] = b

P[2, 11, 7] = b
P[2, 11, 10] = a
P[2, 11, 15] = b

P[2, 12, 8] = b
P[2, 12, 12] = a + b

P[2, 13, 9] = b
P[2, 13, 12] = a
P[2, 13, 13] = b

P[2, 14, 10] = b
P[2, 14, 13] = a
P[2, 14, 14] = b

P[2, 15, 16] = 1
P[2, 16, 16] = 1

# right (a = 3)

P[3, 0, 0] = b
P[3, 0, 1] = a
P[3, 0, 4] = b

P[3, 1, 1] = b
P[3, 1, 2] = a
P[3, 1, 5] = b

P[3, 2, 2] = b
P[3, 2, 3] = a
P[3, 2, 6] = b

P[3, 3, 3] = a + b
P[3, 3, 7] = b

P[3, 4, 0] = b
P[3, 4, 5] = a
P[3, 4, 8] = b

P[3, 5, 1] = b
P[3, 5, 6] = a
P[3, 5, 9] = b

P[3, 6, 2] = b
P[3, 6, 7] = a
P[3, 6, 10] = b

P[3, 7, 3] = b
P[3, 7, 7] = a
P[3, 7, 11] = b

P[3, 8, 4] = b
P[3, 8, 9] = a
P[3, 8, 12] = b

P[3, 9, 5] = b
P[3, 9, 10] = a
P[3, 9, 13] = b

P[3, 10, 6] = b
P[3, 10, 11] = a
P[3, 10, 14] = b

P[3, 11, 7] = b
P[3, 11, 11] = a
P[3, 11, 15] = b

P[3, 12, 8] = b
P[3, 12, 13] = a
P[3, 12, 12] = b

P[3, 13, 9] = b
P[3, 13, 14] = a
P[3, 13, 13] = b

P[3, 14, 10] = b
P[3, 14, 15] = a
P[3, 14, 14] = b

P[3, 15, 16] = 1
P[3, 16, 16] = 1


# Reward function: |A| x |S| array
R = -1 * np.ones([4, 17])

# set rewards
R[:, 15] = 200;  # goal state
R[:, 3] = -80;  # bad state
R[:, 6] = -80;  # bad state
R[:, 9] = -80;  # bad state
R[:, 16] = 0;  # end state

# Discount factor: scalar in [0,1)
discount = 0.9


if __name__ == "__main__":
    main()
