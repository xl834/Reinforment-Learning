from MDP import build_mazeMDP
from DP import DynamicProgramming
import numpy as np
from numpy.testing import assert_allclose
import IL
import visualize


initial_pi = np.array([2, 3, 3, 2, 1, 1, 0, 3, 1, 2, 3, 0, 0, 0, 1, 2, 1])
initial_V_pi = np.array([-58.584607026434036, -240.99917438690397, -293.4241896968443, -334.4514382946084, -94.57320482379257, -167.45960248637064, -311.9075404604963, -181.06114530240242, -92.37900926495321, -177.11857440157402, -136.83415304316443, -154.38281179345748, -87.29656423805085, -120.83520169624771, 26.18175073245016, 200.0, 0.0])


initial_Q_pi = np.array([[-78.35211291745406, -101.02492952978994, -58.584607026434036, -173.5057844635301], [-200.3506674213921, -154.02073712405607, -93.05023730454552, -240.99917438690395], [-263.543072221016, -275.18758320211685, -234.5492634349905, -293.4241896968443], [-375.4676159044495, -278.8317313193596, -334.45143829460835, -360.2986049111998], [-73.28273141352547, -94.57320482379255, -80.9612072383266, -126.87963776575079], [-207.7043804771285, -167.4596024863706, -117.02701512543383, -253.94764657655722], [-311.90754046049625, -213.25581736867795, -243.5844258363147, -252.1533978104147], [-278.2551787035946, -164.8119440078695, -263.49437425200153, -181.06114530240242], [-96.96329283397048, -92.37900926495321, -83.75119466026939, -137.1371206963405], [-216.44332647800937, -187.06995398023196, -177.118574401574, -205.12531498184705], [-242.25443762644193, -29.258184174885653, -151.15768348627785, -136.83415304316443], [-154.38281179345748, 85.68570974705604, -84.64877103301792, -95.70442604570252], [-87.29656423805083, -84.09462387110234, -80.25303789287759, -101.38237949154161], [-120.83520169624771, -85.37667689189215, -96.22059524317795, -24.72925681176233], [-76.51826864618704, 26.181750732450162, -92.06425138058249, 110.06192568805356], [200.0, 200.0, 200.0, 200.0], [0.0, 0.0, 0.0, 0.0]])

# This is the policy pi(s) = argmax_a Q(s,a) run on initial_Q_pi
max_pi = np.array([2, 2, 2, 1, 0, 2, 1, 1, 2, 2, 1, 1, 2, 3, 3, 0, 0])

optimal_pi = np.array([1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 0])

def test_value_function(dp: DynamicProgramming):

    Q_next = np.array([[-58.37900641133568, -67.63882477520329, -55.71039311604798, -77.4237401912582], [-99.19472201415599, -114.30009204131562, -66.26873150470067, -177.1264650420912], [-198.9701017282712, -185.5552307064943, -120.0753354103589, -237.11767663969178], [-324.97042502303384, -253.13795901679504, -287.6579321332199, -315.55588690037246], [-63.60011820941295, -79.45506841872921, -66.383454018226, -93.94235275672827], [-98.30435358746112, -151.26740595858908, -83.64091037084718, -171.82395452259325], [-265.81429544703997, -136.4809155131739, -189.3410249563566, -219.44553015229104], [-227.70313851703048, 1.9428493548113979, -161.4258778545281, -130.90623763721877], [-82.38553961386991, -86.77683269586174, -74.49058149233413, -133.31203072935602], [-168.98328567176924, -110.8356979341562, -151.90034934749121, -117.56975274169945], [-147.69460167062704, 55.99557645511382, -126.51587724987589, 39.05082176376101], [-97.21380877271477, 132.617715952243, -14.682268471240343, 57.73238469958292], [-67.93586242109612, -65.73202365763927, -73.69998526718771, -38.720003186085115], [-108.56050202064286, -12.55523193906151, -78.80887108631329, 41.08955596967334], [4.228894300234125, 92.00056351388584, -5.6709266871326, 135.90850510427768], [200.0, 200.0, 200.0, 200.0], [0.0, 0.0, 0.0, 0.0]])
    VI_final_V = np.array([49.91305784981164, 42.45134802958744, 33.432666787103855, 1.832777857226958, 61.02506641047204, 46.50547429431135, 23.549189941313855, 122.33865427525616, 75.41144881272014, 24.668694861331034, 128.18357979995147, 164.51418480833965, 99.84264586362808, 122.5134744532477, 164.51418480833965, 200.0, 0.0])
    assert_allclose(dp.computeVfromQ(initial_Q_pi, initial_pi), initial_V_pi)
    assert_allclose(dp.computeQfromV(initial_V_pi), initial_Q_pi)
    assert_allclose(dp.extractMaxPifromV(initial_V_pi), max_pi)
    assert_allclose(dp.extractMaxPifromQ(initial_Q_pi), max_pi)
    assert_allclose(dp.valueIterationStep(initial_Q_pi), Q_next)

    [student_pi, student_V, student_nIterations,
        student_epsilon] = dp.valueIteration(initial_Q_pi)
    assert_allclose(student_pi, optimal_pi)
    assert_allclose(student_V, VI_final_V)
    assert_allclose(student_nIterations, 26)
    assert_allclose(student_epsilon, 0.009005268952300582)
    print("Passed Value Iteration Tests!")

def test_policy_evaluation(dp):
    exact_initial_V = np.array([-58.584607026434036, -240.99917438690397, -293.4241896968443, -334.4514382946084, -94.57320482379257, -167.45960248637064, -311.9075404604963, -181.06114530240242, -92.37900926495321, -177.11857440157402, -136.83415304316443, -154.38281179345748, -87.29656423805085, -120.83520169624771, 26.18175073245016, 200.0, 0.0])
    approx_initial_V = np.array([-58.54721936404597, -240.92793088055572, -293.34704660907613, -334.3737114170529, -94.53648978803069, -167.4177880246143, -311.83522926323747, -180.98282297384839, -92.34353145073803, -177.0823860023497, -136.76678641224402, -154.30605045088527, -87.26137476388364, -120.80198704126396, 26.1985089330771, 200.0, 0.0])

    student_approx_V, approx_iters, approx_epsilon = dp.approxPolicyEvaluation(
        initial_pi)

    assert_allclose(student_approx_V, approx_initial_V)
    assert_allclose(approx_iters, 74)
    assert_allclose(approx_epsilon, 0.009063747008696055)
    assert_allclose(dp.exactPolicyEvaluation(initial_pi), exact_initial_V)

    exact_optimal_V = np.array([49.917863946741086, 42.4589656639848, 33.44327288744802, 1.8363183139638797, 61.026971348084274, 46.50926676222191, 23.55024206912671, 122.33900555837026, 75.41203048125863, 24.66899591326577, 128.18369096166154, 164.51421766453677, 99.84300304444099, 122.51360875947864, 164.51421766453677, 200.0, 0.0])
    approx_optimal_V = np.array([49.91163196743455, 42.44909867441332, 33.42955409993253, 1.8317439950647412, 61.0244939884561, 46.504347288552104, 23.548876969819034, 122.33854986525309, 75.41127036154887, 24.66860243493892, 128.18354571195226, 164.51417472035388, 99.84253622942487, 122.51343325611947, 164.51417472035388, 200.0, 0.0])

    student_approx_V, approx_iters, approx_epsilon = dp.approxPolicyEvaluation(
        optimal_pi)

    assert_allclose(dp.exactPolicyEvaluation(optimal_pi), exact_optimal_V)
    assert_allclose(student_approx_V, approx_optimal_V)
    assert_allclose(approx_iters, 27)
    assert_allclose(approx_epsilon, 0.009044308870933548)
    print("Passed policy evaluation tests!")

def test_policy_iteration(dp):
    pi_next = np.array([2, 2, 2, 1, 0, 2, 1, 1, 2, 2, 1, 1, 2, 3, 3, 0, 0])

    assert_allclose(dp.policyIterationStep(initial_pi, True), pi_next)
    assert_allclose(dp.policyIterationStep(initial_pi, False), pi_next)

    exact_optimal_V = np.array([49.917863946741086, 42.4589656639848, 33.44327288744802, 1.8363183139638797, 61.026971348084274, 46.50926676222191, 23.55024206912671, 122.33900555837026, 75.41203048125863, 24.66899591326577, 128.18369096166154, 164.51421766453677, 99.84300304444099, 122.51360875947864, 164.51421766453677, 200.0, 0.0])
    approx_optimal_V = np.array([49.91163196743455, 42.44909867441332, 33.42955409993253, 1.8317439950647412, 61.0244939884561, 46.504347288552104, 23.548876969819034, 122.33854986525309, 75.41127036154887, 24.66860243493892, 128.18354571195226, 164.51417472035388, 99.84253622942487, 122.51343325611947, 164.51417472035388, 200.0, 0.0])

    student_final_exact = dp.policyIteration(initial_pi, True)
    student_final_approx = dp.policyIteration(initial_pi, False)

    assert_allclose(student_final_exact[0], optimal_pi)
    assert_allclose(student_final_exact[1], exact_optimal_V)
    assert_allclose(student_final_exact[2], 6)

    assert_allclose(student_final_approx[0], optimal_pi)
    assert_allclose(student_final_approx[1], approx_optimal_V)
    assert_allclose(student_final_approx[2], 6)
    print("Passed policy iteration tests!")

def test_imitation_learning(dp):
    # (1) Run students' PI code to get expert policy.
    # (2) Run IL.collectData to get training data.
    # (3) Run IL.trainModel to train behavior policy.
    # (4) Run visualize.visualize_IL_state_distribution to generate behavior policy visualization.
    expert_pi, _, _ = dp.policyIteration(initial_pi, True)
    dataset_size = 75
    dataset = IL.collectData(expert_pi, dp, dataset_size)
    assert (len(dataset[0]) == dataset_size)
    
    for i in range(len(dataset)):
        assert_allclose(np.sum(dataset[0][i]), 1)
        assert (dataset[1][i] in range(dp.nActions))

    il_policy = IL.trainModel(dataset[0], dataset[1])

    assert len(il_policy) == len(expert_pi)
    assert ((il_policy.dtype == 'int64') or (il_policy.dtype == 'int32'))

    visualizer = visualize.Visualize(dp)

    #visualizer.visualize_IL_policy(max_pi)
    visualizer.visualize_IL_policy(il_policy)
    #visualizer.visualize_IL_policy(optimal_pi)
    print("Passed imitation learning tests!")

'''
Where we test all methods: value iteration, policy iteration, and policy evaluation.
'''


def main():
    mdp = build_mazeMDP()
    dp = DynamicProgramming(mdp)
    test_value_function(dp)
    test_policy_evaluation(dp)
    test_policy_iteration(dp)
    test_imitation_learning(dp)


if __name__ == '__main__':
    main()
