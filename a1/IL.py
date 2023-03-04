## Code for Imitation Learning
from sklearn.linear_model import LogisticRegression
import numpy as np

def collectData(expert_policy, dp, sample_size=500):
    #initialize dataset
    states = []
    actions = []
    
    #initialize state
    s = 0
    
    # Main data collection loop
    for i in range(sample_size):
        state = np.zeros(17)
        action = expert_policy[s]
        prob = dp.P[action][s]
        prob_idx = np.nonzero(prob)[0]
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
    X = states
    y = actions
    # Learn policy using logistic regression
    clf = LogisticRegression(random_state=0).fit(X, y)

    # Convert policy to vector form
    # Note that in collectData, we collected a dataset (s,a), so when we
    # train using logistic regression, we'll get a model phi that maps a one-hot vector s to
    # an integer a. However, our policy evaluation code requires a vector.
    pi = np.zeros(len(states[0]))
    for s in range(len(states[0])):
        one_hot_s = np.zeros(states[0].shape[0])
        one_hot_s[s] = 1
        one_hot_s = one_hot_s.reshape((1, 17))
        pi[s] = int(clf.predict(one_hot_s))
    pi = pi.astype(int)
    return pi